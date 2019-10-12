import time
import os
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


def main():
    lipschitz_constants = []
    start_time = time.time()
    # entropy_avg_meter = AverageValueMeter()
    # latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()

    for epoch in range(args.begin_epoch, args.epochs):
        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_count = 0, 0
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            step = bidx + len(train_loader) * epoch
            model.train()
            if args.random_rotate:
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis)
            inputs = tr_batch.to(device)
            out, loss = model(inputs, step, writer)
            train_count += 1
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_lipschitz(model)

            #entropy, prior_nats, recon_nats = out['entropy'], out['prior_nats'], out['recon_nats']
            recon_nats = out['recon_nats']
            #entropy_avg_meter.update(entropy)
            point_nats_avg_meter.update(recon_nats)
            #latent_nats_avg_meter.update(prior_nats)
            # if step % args.log_freq == 0:
            #     duration = time.time() - start_time
            #     start_time = time.time()
            #     print("Epoch %d Batch [%2d/%2d] Time [%3.2fs]  PointNats %2.5f"
            #           % (epoch, bidx, len(train_loader), duration, point_nats_avg_meter.avg))

        scheduler.step()
        lipschitz_constants.append(get_lipschitz_constants(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        writer.add_scalar('avg_train_loss', train_loss / train_count, epoch)
        print("Epoch %d Time [%3.2fs]  Likelihood Loss  %2.5f"
                      % (epoch, time.time() - start_time, train_loss / train_count))

        # generate samples
        if epoch % args.val_freq == 0:
            print('Start testing the model at epoch {}'.format(epoch))
            model.eval()
            with torch.no_grad():
                _, samples = model.sample(args.val_batchsize, args.tr_max_sample_points, truncate_std=None, gpu=device)
            test_path = os.path.join('checkpoints', args.save, 'test_results/')
            if not os.path.isdir(test_path):
                os.mkdir(test_path)
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

    #init_layer = layers.LogitTransform(0.05)
    init_layer = layers.TanhTransform()
    model = PCResFlow_1(args, input_size, init_layer=init_layer)
    model.to(device)
    ema = utils.ExponentialMovingAverage(model)
    logger.info(model)
    logger.info('EMA: {}'.format(ema))

    ######################################
    #########  optimizer   ###############
    ######################################
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        if args.scheduler:
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
