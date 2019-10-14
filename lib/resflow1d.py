import numpy as np
import torch
import torch.nn as nn

import lib.layers as layers
import lib.layers.base as base_layers

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
    'tanh': lambda b: nn.Tanh(),
}


class ResidualFlow1d(nn.Module):

    def __init__(
        self,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        squeeze=True,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=None,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='tanh',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        block_type='resblock',
    ):
        super(ResidualFlow1d, self).__init__()
        self.n_scale = len(n_blocks)
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.squeeze = squeeze
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.block_type = block_type

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)
        self.transforms = self._build_net(input_size)
        self.dims = self.calc_output_size(input_size)

    def _build_net(self, input_size):
        _, c, l = input_size
        transforms = []
        _stacked_blocks = StackediResBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    initial_size=(c, l),
                    idim=self.intermediate_dim,
                    last_block = (i == self.n_scale - 1),
                    squeeze = self.squeeze,  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None, # only have init_layer in the beginning
                    n_blocks=self.n_blocks[i],
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    n_lipschitz_iters=self.n_lipschitz_iters,
                    sn_atol=self.sn_atol,
                    sn_rtol=self.sn_rtol,
                    n_power_series=self.n_power_series,
                    n_dist=self.n_dist,
                    n_samples=self.n_samples,
                    kernels=self.kernels,
                    activation_fn=self.activation_fn,
                    fc_end=self.fc_end,
                    fc_idim=self.fc_idim,
                    n_exact_terms=self.n_exact_terms,
                    preact=self.preact,
                    neumann_grad=self.neumann_grad,
                    grad_in_forward=self.grad_in_forward,
                    first_resblock=self.first_resblock and (i == 0),
                )
            )
            # update the initial size
            if self.squeeze:  # if add squeeze layers
                c, l = c * 2, l // 2
        return nn.ModuleList(transforms)

    def calc_output_size(self, input_size):
        n, c, l = input_size
        if self.squeeze:
            k = self.n_scale - 1
            return [[c * 2**k, l // 2**k]]
        else:
            return [[c, l]]


    def forward(self, x, logpx=None, inverse=False):
        if inverse:
            return self.inverse(x, logpx)
        out = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(x, logpx)
            else:
                x = self.transforms[idx].forward(x)

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)  # (batch_size, c*l)
        output = out if logpx is None else (out, logpx)
        return output

    def inverse(self, z, logpz=None):
        z = z.view(z.shape[0], *self.dims[-1])
        for idx in range(len(self.transforms) - 1, -1, -1):
            if logpz is None:
                z = self.transforms[idx].inverse(z)
            else:
                z, logpz = self.transforms[idx].inverse(z, logpz)
        return z if logpz is None else (z, logpz)


class StackediResBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        last_block = False,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=1,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
    ):

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1]))
            else:
                # need to check how to do actnorm for 1d conv, size[0]: #channels
                return layers.ActNormconv1d(size[0])

        def _lipschitz_layer(fc):
            return base_layers.get_linear1d if fc else base_layers.get_conv1d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                return layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,  # (c, l)
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
                ks = list(map(int, kernels.split('-'))) # kernal size [3, 1, 3] by default
                nnet = []
                '''
                architecture:
                batchnorm 
                preact 
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
                #if dropout: nnet.append(nn.Dropout(dropout, inplace=True))  #no 1d dropout, need to configure later
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

        # initialize the network architecture

        if init_layer is not None: chain.append(init_layer)
        # add different act norm depends on mlp or conv1d
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if not last_block:
            '''
            resblock -> actnorm -> resblock -> actnorm ...
            '''
            for i in range(n_blocks):
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            if squeeze:
                print('add squueze layers....')
                chain.append(layers.SqueezeLayer1d(2))
        else:
            for _ in range(n_blocks):
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                print('add {} fc blocks in the end '.format(fc_nblocks))
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, n_iterations, activation_fn,
            preact, dropout, sn_atol, sn_rtol, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, l = self.input_shape
        dim = c * l
        nnet = []
        last_dim = dim // div_in
        if preact: nnet.append(ACT_FNS[activation_fn](False))
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations,
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout: nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations,
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        y = self.nnet(x)
        return y.view(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.view(*shape), logpy

    def inverse(self, y, logpy=None):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logpx = self.fc_module.inverse(y, logpy)
            return x.view(*shape), logpx

def pc_resflow(args, input_size, init_layer=None):

    model = ResidualFlow1d(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim, # intermediate dimension of the convnet
        squeeze=args.squeeze,  # factor out half of the channels
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        n_lipschitz_iters=args.n_lipschitz_iters,
        sn_atol=args.sn_tol,
        sn_rtol=args.sn_tol,
        n_power_series=args.n_power_series,
        n_dist=args.n_dist,
        n_samples=args.n_samples,
        kernels=args.kernels,
        activation_fn=args.act,
        fc_end=args.fc_end,
        fc_idim=args.fc_idim,
        n_exact_terms=args.n_exact_terms,
        preact=args.preact,
        neumann_grad=args.neumann_grad,
        grad_in_forward=args.mem_eff,
        first_resblock=args.first_resblock,
        block_type=args.block,
    )

    return model

def latent_resflow(args, input_size, init_layer=None):

    model = ResidualFlow1d(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim,
        squeeze=args.squeeze,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        n_lipschitz_iters=args.n_lipschitz_iters,
        sn_atol=args.sn_tol,
        sn_rtol=args.sn_tol,
        n_power_series=args.n_power_series,
        n_dist=args.n_dist,
        n_samples=args.n_samples,
        kernels=args.kernels,
        activation_fn=args.act,
        fc_end=args.fc_end,
        fc_idim=args.fc_idim,
        n_exact_terms=args.n_exact_terms,
        preact=args.preact,
        neumann_grad=args.neumann_grad,
        grad_in_forward=args.mem_eff,
        first_resblock=args.first_resblock,
        block_type=args.block,
    )

    return model