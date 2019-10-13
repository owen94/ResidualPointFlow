import argparse
import numpy as np
from lib.resflow1d import ACT_FNS
def init_parse():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')
    parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
    parser.add_argument('--nblocks', type=str, default='1-1')
    parser.add_argument('--squeeze', type=eval, choices=[True, False], default=False)
    parser.add_argument('--idim', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--init-layer', type=str, choices=['norm','tanh','none'], default='none'),


    parser.add_argument('--coeff', type=float, default=0.98)
    parser.add_argument('--n-lipschitz-iters', type=int, default=None)
    parser.add_argument('--sn-tol', type=float, default=1e-3)

    parser.add_argument('--n-power-series', type=int, default=None)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--n-exact-terms', type=int, default=2)
    parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
    parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

    parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='tanh')
    parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--fc', type=eval, default=False, choices=[True, False]) #use fc or conv1d in resblock
    parser.add_argument('--kernels', type=str, default='3-1-3')
    parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
    parser.add_argument('--fc-idim', help='fc dims',type=int, default=512)
    parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
    parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')

    # data parameters
    parser.add_argument(
        '--dataset_type', type=str, default='shapenet15k', choices=[
            'shapenet15k',
        ]
    )
    parser.add_argument('--cates', type=list, default=['airplane'])
    parser.add_argument('--tr_max_sample_points', type=int, default=2048)
    parser.add_argument('--te_max_sample_points', type=int, default=2048)
    parser.add_argument('--dataset_scale', type=float, default=1.)
    parser.add_argument('--data_dir', type=str, default="data/ShapeNetCore.v2.PC15k")
    parser.add_argument('--normalize_per_shape', type=bool, default=False)
    parser.add_argument('--normalize_std_per_axis', type=bool, default=True)
    parser.add_argument('--random_rotate', type=bool, default=True)


    # optimizer parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument('--epochs', help='Number of epochs for training', type=int, default=1000)
    parser.add_argument('--batch_size', help='Minibatch size', type=int, default=32)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=5e-4)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--annealing-iters', type=int, default=0)

    # train
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--begin-epoch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=False)

    # save and eval
    parser.add_argument('--save', help='directory to save results', type=str, default='exp1')
    parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=50)
    parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
    parser.add_argument('--update-freq', type=int, default=1)
    parser.add_argument('--log-freq', type=int, default=20)
    parser.add_argument('--val-freq', type=int, default=20)

    # others
    parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
    parser.add_argument('--var-reduc-lr', type=float, default=0)

    args = parser.parse_args()
    # Random seed
    if args.seed is None:
        args.seed = np.random.randint(100000)

    return args