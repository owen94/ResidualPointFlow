import argparse
import numpy as np

def init_parse():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', help='Minibatch size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='cd', choices=['cd'])
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()
    # Random seed
    if args.seed is None:
        args.seed = np.random.randint(100000)

    return args