import os, time, glob
import torch
import numpy as np
from model import Transfomer
from datasets import get_datasets, init_np_seed
from args import init_parse
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from loss import get_loss
def apply_random_rotation(pc, rot_axis=1):
    B = pc.shape[0]

    theta = np.random.rand(B) * 2 * np.pi
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    if rot_axis == 0:
        rot = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 1:
        rot = np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 2:
        rot = np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ]).T.reshape(B, 3, 3)
    else:
        raise Exception("Invalid rotation axis")
    rot = torch.from_numpy(rot).to(pc)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    pc_rotated = torch.bmm(pc, rot)
    return pc_rotated, rot, theta

def main():
    for epoch in range(args.begin_epoch, args.epochs):
        train_loss, train_count = 0, 0
        model.train()
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            tr_batch, _, _ = apply_random_rotation(tr_batch, rot_axis=train_loader.dataset.gravity_axis)
            input1 = tr_batch.to(device)
            input2 = torch.randn(size=(args.batch_size, 3))
            recon = model(input1, input2)
            loss = loss_criterion(recon, te_batch)
            train_count += 1
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        writer.add_scalar('avg_train_loss', train_loss / train_count, epoch)
        print('Epoch {}: Charmer Distance {}'.format(epoch, train_loss/train_count))


if __name__ == '__main__':
    # arguments and logger
    args = init_parse()

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    tr_dataset, te_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    model = Transfomer(zdim=128)
    model.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
        )
    else:
        raise ValueError('Unknown optimizer {}'.format(args.optimizer))

    loss_criterion = get_loss(args.loss, args.alpha)


    if args.save is not None:
        log_dir = "runs/%s" % args.save
    else:
        log_dir = "runs/time-%d" % time.time()
    writer = SummaryWriter(logdir=log_dir)

    model_path = os.path.join('checkpoints', args.save, 'models')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    main()
