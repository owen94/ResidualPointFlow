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
}


class ResidualFlow1d(nn.Module):

    def __init__(
        self,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=True,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        n_classes=10,
        block_type='resblock',
    ):
        super(ResidualFlow1d, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
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
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.n_classes = n_classes
        self.block_type = block_type

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        if self.classification:
            raise ValueError('Classification not implemented ')
            #self.build_multiscale_classifier(input_size)

    def _build_net(self, input_size):
        _, c, l = input_size
        transforms = []
       # _stacked_blocks = StackediResBlocks if self.block_type == 'resblock' else StackedCouplingBlocks
        _stacked_blocks = StackediResBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    initial_size=(c, l),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    quadratic=self.quadratic,
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    vnorms=self.vnorms,
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
                    learn_p=self.learn_p,
                )
            )
            # update the initial size
            c, l = c * 2 if self.factor_out else c * 2, l // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, l = input_size
        n_scale = 0
        while l >= 256:
            n_scale += 1
            l = l // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, l = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 2**k, l // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                l = l // 2
                output_sizes.append((n, c, l))
            else:
                output_sizes.append((n, c, l))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, inverse=False, classify=False):
        if inverse:
            return self.inverse(x, logpx)
        out = []
        if classify: class_outs = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(x, logpx)
            else:
                x = self.transforms[idx].forward(x)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)

            # Handle classification.
            if classify:
                if self.factor_out:
                    class_outs.append(self.classification_heads[idx](f))
                else:
                    class_outs.append(self.classification_heads[idx](x))

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        if classify:
            h = torch.cat(class_outs, dim=1).squeeze(-1).squeeze(-1)
            logits = self.logit_layer(h)
            return output, logits
        else:
            return output

    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logpz)
                return z_prev, logpz
        else:
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
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        chain = []

        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1]))
            else:
                # need to check how to do actnorm for 1d conv, size[0]: #channels
                return layers.ActNormconv1d(size[0])

        # disable this in 3d point cloud for now
        def _quadratic_layer(initial_size, fc):
            if fc:
                c, l = initial_size
                dim = c * l
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv1d(initial_size[0])

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
                if learn_p:
                    _domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(ks))]
                    _codomains = _domains[1:] + [_domains[0]]
                else:
                    _domains = domains
                    _codomains = codomains
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
                        domain=_domains[0], codomain=_codomains[0], atol=sn_atol, rtol=sn_rtol
                    )
                )
                if batchnorm: nnet.append(layers.MovingBatchNormconv1d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(
                        _lipschitz_layer(fc)(
                            idim, idim, k, 1, k // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[i + 1], codomain=_codomains[i + 1], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm: nnet.append(layers.MovingBatchNormconv1d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                #if dropout: nnet.append(nn.Dropout(dropout, inplace=True))  #no 1d dropout, need to configure later
                nnet.append(
                    _lipschitz_layer(fc)(
                        idim, initial_size[0], ks[-1], 1, ks[-1] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                        domain=_domains[-1], codomain=_codomains[-1], atol=sn_atol, rtol=sn_rtol
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
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if squeeze:
            '''
            resblock -> actnorm -> resblock -> actnorm ...
            '''
            print('add squueze layers....')
            c, l = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer1d(2))
        else:
            print('Donot add squeeze layers but add fc blocks in the end ')
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
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
        x = x.view(x.shape[0], -1)
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

def pc_resflow(args, input_size, init_layer=None, n_classes=None):

    model = ResidualFlow1d(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim, # intermediate dimension of the convnet
        factor_out=args.factor_out,  # factor out half of the channels
        quadratic=args.quadratic,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        vnorms=args.vnorms,
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
        learn_p=args.learn_p,
        classification=args.task in ['classification', 'hybrid'],
        classification_hdim=args.cdim,
        n_classes=n_classes,
        block_type=args.block,
    )

    return model

def latent_resflow(args, input_size, init_layer=None, n_classes=None):

    model = ResidualFlow1d(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim,
        factor_out=args.factor_out,
        quadratic=args.quadratic,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        vnorms=args.vnorms,
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
        learn_p=args.learn_p,
        classification=args.task in ['classification', 'hybrid'],
        classification_hdim=args.cdim,
        n_classes=n_classes,
        block_type=args.block,
    )

    return model