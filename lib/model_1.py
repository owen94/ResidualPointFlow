import torch
from torch import nn
from lib.resflow1d import pc_resflow
from utils import truncated_normal, standard_normal_logprob

# Model
class PCResFlow_1(nn.Module):
    def __init__(self, args, input_size):
        super(PCResFlow_1, self).__init__()
        self.input_dim = args.input_dim
        self.distributed = args.distributed
        self.truncate_std = None
        self.point_rsf = pc_resflow(args, input_size)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    def forward(self, x, step, writer=None):
        # x is (n, l, c)
        batch_size = x.size(0)
        num_points = x.size(1)

        x = x.transpose(1, 2)
        # Compute the reconstruction likelihood P(X|z)
        y, delta_log_py = self.point_rsf(x, 0)
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        log_px = log_py - delta_log_py

        # Loss
        loss = -log_px.mean()
        recon_nats = loss / float(x.size(1) * x.size(2))

        if writer is not None:
            writer.add_scalar('train/recon', loss, step)
            writer.add_scalar('train/recon_nats', recon_nats, step)

        return {
            'recon_nats': recon_nats,
        }, loss

    def sample(self, batch_size, num_points, truncate_std=None, gpu=None):
        # Generate the shape code from the prior
        z = self.sample_gaussian((batch_size, num_points * self.input_dim), truncate_std, gpu=gpu)
        # Sample points conditioned on the shape code
        x = self.point_rsf(z, reverse=True)
        return z, x

