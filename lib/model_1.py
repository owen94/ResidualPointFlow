import torch
from torch import nn
from lib.resflow1d import pc_resflow
from utils import truncated_normal, standard_normal_logprob

# Model
class PCResFlow_1(nn.Module):
    def __init__(self, args, input_size, init_layer=None):
        super(PCResFlow_1, self).__init__()
        self.input_dim = args.input_dim
        self.distributed = args.distributed
        self.truncate_std = None
        self.point_rsf = pc_resflow(args, input_size, init_layer)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.to(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    def forward(self, x, step, writer=None):
        # x is (n, l, c)
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        # Compute the reconstruction likelihood P(X|z)
        y, delta_log_py = self.point_rsf(x, 0)
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        log_px = log_py - delta_log_py

        # Loss
        loss = -log_px.mean() / (float(x.size(1) * x.size(2)))
        recon_nats = loss.item()

        if writer is not None:
            writer.add_scalar('train/recon', loss.item(), step)
            writer.add_scalar('train/recon_nats', recon_nats, step)

        return loss, log_py.mean().detach(), delta_log_py.mean().detach()

    def sample(self, batch_size, num_points, truncate_std=None, gpu=None):
        # Generate the shape code from the prior
        z = self.sample_gaussian((batch_size, num_points * self.input_dim), truncate_std, gpu=gpu)
        # Sample points conditioned on the shape code
        x = self.point_rsf(z, inverse=True)
        return z, x

