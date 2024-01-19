from torch import nn
from collections import OrderedDict


class Critic(nn.Module):
    def __init__(self, ninput, noutput, layer_sizes, batch_norm=True):
        super().__init__()
        layers = []

        inp_size = ninput
        for idx, layer_size in enumerate(layer_sizes):
            layer = nn.Linear(inp_size, layer_size)
            layers.append(("linear_{}".format(idx), layer))

            if idx > 0 and batch_norm is True:
                bn = nn.BatchNorm1d(layer_size, eps=1e-05, momentum=0.1)
                layers.append(("bn_{}".format(idx), bn))

            layers.append(("activation_{}".format(idx), nn.LeakyReLU(0.2)))
            inp_size = layer_size

        layer = nn.Linear(inp_size, noutput)
        layers.append(("linear_last", layer))

        self.layers = nn.Sequential(OrderedDict(layers))

        self.init_weights()

    @classmethod
    def from_opts(cls, opts):
        arch_d = opts.get("arch_d")
        layer_sizes = list(map(int, arch_d.split("-")))
        return cls(opts["ae_hidden_size"], 1, layer_sizes)

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def forward(self, inp, return_last=False):
        activations = []
        activation_ = None
        for layer in self.layers:
            activation_ = layer(inp)
            activations.append(activation_)

        if return_last:
            return activation_, activations[-2]
        else:
            return activation_
