import torch
import torch.nn as nn

__all__ = ['SqueezeLayer', 'SqueezeLayer1d']


class SqueezeLayer(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def inverse(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze(input, upscale_factor=2):
    return torch.pixel_shuffle(input, upscale_factor)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.reshape(batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    output = input_view.permute(0, 1, 3, 5, 2, 4)
    return output.reshape(batch_size, out_channels, out_height, out_width)


# implement for 1D convolution

class SqueezeLayer1d(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer1d, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None):
        squeeze_x = squeeze1d(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def inverse(self, y, logpy=None):
        unsqueeze_y = unsqueeze1d(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze1d(input, upscale_factor=2):
    '''
       [:, C*r, L] -> [:, C, L*r]
       '''
    batch_size, in_channels, length = input.shape
    out_channels = in_channels // upscale_factor
    out_len = length * upscale_factor
    input_view = input.reshape(batch_size, out_channels, upscale_factor, length)
    output = input_view.permute(0, 1, 3, 2)
    return output.reshape(batch_size, out_channels, out_len)

def squeeze1d(input, downscale_factor=2):
    '''
    [:, C, L*r] -> [:, C*r, L]
    '''
    batch_size, in_channels, length = input.shape
    out_channels = in_channels * downscale_factor
    out_len = length // downscale_factor
    input_view = input.reshape(batch_size, in_channels, out_len, downscale_factor)
    output = input_view.permute(0, 1, 3, 2)
    return output.reshape(batch_size, out_channels, out_len)

# import torch
# inputs = torch.randn(1, 2, 4)
# print(inputs)
# output = squeeze1d(inputs)
# print(output)
# new_input = unsqueeze1d(output)
# print(new_input)