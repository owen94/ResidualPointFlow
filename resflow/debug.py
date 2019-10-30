'''
test building blocks in resflow 1d models
we test the model at three levels:
- spectrum norm 1d conv
- invertible resibual block
- residual flow 1d model
'''

import lib.layers as layers
import lib.layers.base as base_layers
import torch
import torch.nn as nn
from lib.resflow1d import FCNet, pc_resflow
from args import init_parse


ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
}

def test_conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=True):
    model = base_layers.get_conv1d( in_channels, out_channels, kernel_size, stride, padding, bias,atol=1e-3, rtol=1e-3)
    x = torch.randn(2, 4, 10, dtype=torch.float32)
    output = model(x)
    print('output size:', output.size())
    repr = model.extra_repr()
    print(repr)



def _lipschitz_layer(fc):
    return base_layers.get_linear1d if fc else base_layers.get_conv1d

def get_iresblock(initial_size, fc, idim, first_resblock=False, n_power_series=5, kernels='3-1-3',
                  batchnorm=True, preact=False, activation_fn='relu', n_lipschitz_iters=2000, coeff=0.97,
                  sn_atol=1e-3, sn_rtol=1e-3, dropout=True, n_dist='geometric',
                  n_samples=1,n_exact_terms=0,neumann_grad=True, grad_in_forward=False):
    if fc:
        return layers.iResBlock(
            FCNet(
                input_shape=initial_size,
                idim=idim,
                lipschitz_layer=_lipschitz_layer(True),
                nhidden=len(kernels.split('-')) - 1,
                coeff=coeff,
                n_iterations=n_lipschitz_iters,
                activation_fn=activation_fn,
                preact=preact,
                dropout=dropout,
                sn_atol=sn_atol,
                sn_rtol=sn_rtol,
            ),
            n_power_series=n_power_series,
            n_dist=n_dist,
            n_samples=n_samples,
            n_exact_terms=n_exact_terms,
            neumann_grad=neumann_grad,
            grad_in_forward=grad_in_forward,
        )


    else:
        ks = list(map(int, kernels.split('-')))  # kernal size [3, 1, 3] by default
        nnet = []
        '''
        architecture:
        batchnorm 
        conv1d 
        batchnorm 
        conv1d 
        batchnorm 
        dropout 
        conv1d 
        batchnorm
    
        '''
        if not first_resblock and preact:
            if batchnorm: nnet.append(layers.MovingBatchNormconv1d(initial_size[0]))
            nnet.append(ACT_FNS[activation_fn](False))
        nnet.append(
            _lipschitz_layer(fc)(
                initial_size[0], idim, ks[0], 1, ks[0] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
               atol=sn_atol, rtol=sn_rtol
            )
        )
        if batchnorm: nnet.append(layers.MovingBatchNormconv1d(idim))
        nnet.append(ACT_FNS[activation_fn](True))
        for i, k in enumerate(ks[1:-1]):
            nnet.append(
                _lipschitz_layer(fc)(
                    idim, idim, k, 1, k // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            if batchnorm: nnet.append(layers.MovingBatchNormconv1d(idim))
            nnet.append(ACT_FNS[activation_fn](True))
        if dropout: nnet.append(nn.Dropout2d(dropout, inplace=True))
        nnet.append(
            _lipschitz_layer(fc)(
                idim, initial_size[0], ks[-1], 1, ks[-1] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                 atol=sn_atol, rtol=sn_rtol
            )
        )
        if batchnorm: nnet.append(layers.MovingBatchNormconv1d(initial_size[0]))
        return layers.iResBlock(
            nn.Sequential(*nnet),
            n_power_series=n_power_series,
            n_dist=n_dist,
            n_samples=n_samples,
            n_exact_terms=n_exact_terms,
            neumann_grad=neumann_grad,
            grad_in_forward=grad_in_forward,
        )

def test_iresblock(initial_size, fc, idim, first_resblock=False, n_power_series=5, kernels='3-1-3',
                  batchnorm=True, preact=False, activation_fn='relu', n_lipschitz_iters=2000, coeff=0.97,
                  sn_atol=1e-3, sn_rtol=1e-3, dropout=False, n_dist='geometric',
                  n_samples=1,n_exact_terms=0,neumann_grad=True, grad_in_forward=False):
    model = get_iresblock(initial_size, fc, idim, first_resblock=False, n_power_series=5, kernels='3-1-3',
                  batchnorm=True, preact=False, activation_fn='relu', n_lipschitz_iters=2000, coeff=0.97,
                  sn_atol=1e-3, sn_rtol=1e-3, dropout=False, n_dist='geometric',
                  n_samples=1,n_exact_terms=0,neumann_grad=True, grad_in_forward=False)
    for p in model.parameters():
        print('parameter size in resblock:', p.size())
    x = torch.randn(2, 4, 10, dtype=torch.float32)
    output, logp = model(x, 0)
    print('output size:', output.size())
    print('log determinant', logp)


args = init_parse()
def test_resflow(args, input_size):
    model = pc_resflow(args, input_size, init_layer=None, n_classes=None)
    for p in model.parameters():
        print(p.size())
    x = torch.randn(2, 4, 1024, dtype=torch.float32)
    output, logp = model(x, 0)
    print('output size:', output.size())
    print('log determinant', logp)

#test_conv1d(4, 8, 3, 1, 1)
#test_iresblock([4, 10], True, 64)  # using linear model
test_iresblock([4, 10], False, 64) # using conv1d model
#test_resflow(args, [2, 4, 1024] )


