import time
import os, glob
import os.path
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
import lib.optimizers as optim
import lib.utils as utils
from lib.lr_scheduler import CosineAnnealingWarmRestarts
from datasets import get_datasets, init_np_seed
from lib.model import PCResFlow
from lib.model_1 import PCResFlow_1
from args import init_parse
from utils import *
from tensorboardX import SummaryWriter
from lib.resflow1d import ACT_FNS


def main():
    lipschitz_constants = []
    start_time = time.time()
    # entropy_avg_meter = AverageValueMeter()
    # latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    logpz_meter = utils.RunningAverageMeter(0.93)
    delta_logp_meter = utils.RunningAverageMeter(0.93)
    end = time.time()

    for epoch in range(args.begin_epoch, args.epochs):
        train_loss, train_count = 0, 0
        train_logpz, train_delta_logp = 0, 0
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            step = bidx + len(train_loader) * epoch
            model.train()
            if args.random_rotate:
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis)
            # use toy model
            # inputs = tr_batch.view(-1, args.tr_max_sample_points*3).to(device)
            # zero = torch.zeros(inputs.shape[0], 1).to(inputs)
            # # transform to z
            # z, delta_logp = model(inputs, zero)
            #
            # #compute log p(z)
            # logpz = standard_normal_logprob(z).sum(1, keepdim=True)
            # logpx = logpz - delta_logp
            # loss = -torch.mean(logpx)/args.tr_max_sample_points/3/np.log(2)
            #
            inputs = tr_batch.to(device)
            loss, logpz, delta_logp = model(inputs,step, writer)

            loss_meter.update(loss.item())
            logpz_meter.update(torch.mean(logpz).item())
            delta_logp_meter.update(torch.mean(-delta_logp).item())

            train_count += 1
            train_loss += loss.item()
            train_logpz += torch.mean(logpz).item()
            train_delta_logp += torch.mean(-delta_logp).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update_lipschitz(model, 5)
            update_lipschitz(model)


        #scheduler.step()
        lipschitz_constants.append(get_lipschitz_constants(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        writer.add_scalar('avg_train_loss', train_loss / train_count, epoch)
        writer.add_scalar('avg_lopz', train_logpz / train_count, epoch)
        writer.add_scalar('avg_neg_deltalogz', train_delta_logp / train_count, epoch)

        # print("Epoch %d Time [%3.2fs]  Likelihood Loss  %2.5f"
        #               % (epoch, time.time() - start_time, train_loss / train_count))

        time_meter.update(time.time() - end)
        logger.info(
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'
            ' | Logp(z) {:.6f}({:.6f}) | DeltaLogp {:.6f}({:.6f})'.format(
                epoch, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, logpz_meter.val, logpz_meter.avg,
                delta_logp_meter.val, delta_logp_meter.avg
            )
        )

        # generate samples
        if epoch % args.val_freq == 0:
            print('Start testing the model at epoch {}'.format(epoch))
            model.eval()
            with torch.no_grad():
                _, samples = model.sample(args.val_batchsize, args.tr_max_sample_points, truncate_std=None, gpu=device)
                #samples = model.inverse(torch.randn(args.val_batchsize, args.tr_max_sample_points*3).to(device))
            test_path = os.path.join('checkpoints', args.save, 'test_results/')
            if not os.path.isdir(test_path):
                os.mkdir(test_path)
            elif epoch == 0:
                files = glob.glob(test_path + '*.npy')
                for f in files:
                    os.remove(f)
            np.save(os.path.join(test_path, 'samples_' + str(epoch) + '.npy'), samples.detach().cpu().numpy())

            # save the recent model (should save the best one)
            torch.save(model.state_dict(), os.path.join('checkpoints', args.save, 'models/model.t7'))

if __name__ == '__main__':

    ######################################
    #########  set up      ###############
    ######################################
    # arguments and logger
    args = init_parse()
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # cuda device
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    else:
        logger.info('WARNING: Using device {}'.format(device))

    # random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Dataset and hyperparameters
    tr_dataset, te_dataset = get_datasets(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    ######################################
    #########    model     ###############
    ######################################
    logger.info('Dataset loaded.')
    logger.info('Creating model.')
    n_channels = 3  # x, y, z
    input_size = (args.batch_size, n_channels, args.tr_max_sample_points)
    dataset_size = len(train_loader.dataset)

    if args.init_layer == 'norm':
        init_layer = layers.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    elif args.init_layer == 'none':
        init_layer = None
    elif args.init_layer == 'tanh':
        init_layer = layers.TanhTransform()
    model = PCResFlow_1(args, input_size, init_layer=init_layer)

    # ACTIVATION_FNS = {
    #     'relu': torch.nn.ReLU,
    #     'tanh': torch.nn.Tanh,
    #     'elu': torch.nn.ELU,
    #     'selu': torch.nn.SELU,
    #     'fullsort': base_layers.FullSort,
    #     'maxmin': base_layers.MaxMin,
    #     'swish': base_layers.Swish,
    #     'lcube': base_layers.LipschitzCube,
    # }
    #
    # def build_nnet(dims, activation_fn=torch.nn.ReLU):
    #     nnet = []
    #     for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:],)):
    #         nnet.append(activation_fn())
    #         nnet.append(
    #             base_layers.get_linear1d(
    #                 in_dim,
    #                 out_dim,
    #                 coeff=args.coeff,
    #                 n_iterations=5,
    #                 atol=None,
    #                 rtol=None,
    #                 zero_init=(out_dim == args.tr_max_sample_points * 3),
    #             )
    #         )
    #     return torch.nn.Sequential(*nnet)
    #
    #
    # activation_fn = ACTIVATION_FNS['swish']
    #
    # dims = [args.tr_max_sample_points * 3] + list(map(int, '128-128-128-128'.split('-'))) + [args.tr_max_sample_points * 3]
    # blocks = []
    # if args.actnorm: blocks.append(layers.ActNorm1d(2))
    # for _ in range(100):
    #     blocks.append(
    #         layers.iResBlock(
    #             build_nnet(dims, activation_fn),
    #             n_dist=args.n_dist,
    #             n_power_series=args.n_power_series,
    #             exact_trace=False,
    #             brute_force=False,
    #             n_samples=args.n_samples,
    #             neumann_grad=False,
    #             grad_in_forward=False,
    #         )
    #     )
    #     if args.actnorm: blocks.append(layers.ActNorm1d(2))
    #     if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
    # model = layers.SequentialFlow(blocks).to(device)

    model.to(device)
    ema = utils.ExponentialMovingAverage(model)
    logger.info(model)
    #logger.info('EMA: {}'.format(ema))

    ######################################
    #########  optimizer   ###############
    ######################################
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
            )
    else:
        raise ValueError('Unknown optimizer {}'.format(args.optimizer))
    logger.info(optimizer)

    ######################################
    #########  writter     ###############
    ######################################
    if args.save is not None:
        log_dir = "runs/%s" % args.save
    else:
        log_dir = "runs/time-%d" % time.time()
    writer = SummaryWriter(logdir=log_dir)

    model_path = os.path.join('checkpoints', args.save, 'models')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    main()
