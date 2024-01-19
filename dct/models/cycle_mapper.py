from torch import nn
from collections import OrderedDict
import torch


class CycleMapper(nn.Module):
    def __init__(self, ninput, noutput, layer_sizes, only_linear):
        super().__init__()
        layers = []

        inp_size = ninput
        for idx, layer_size in enumerate(layer_sizes):
            layer = nn.Linear(inp_size, layer_size)
            layers.append(("linear_{}".format(idx), layer))

            # add non linearity if the flag is set
            if not only_linear:
                layers.append(("activation_{}".format(idx), nn.ReLU()))
            inp_size = layer_size

        layer = nn.Linear(inp_size, noutput)
        layers.append(("linear_last", layer))

        self.layer = nn.Sequential(OrderedDict(layers))

        self.init_weights()

    @classmethod
    def from_opts(cls, opts):
        arch_d = opts.get("arch_mapper")
        layer_sizes = list(map(int, arch_d.split("-")))
        return cls(
            opts["ae_hidden_size"],
            opts["ae_hidden_size"],
            layer_sizes,
            opts["cycle_mapper_only_linear"],
        )

    def init_weights(self):
        init_std = 0.02
        for layer in self.layer:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def forward(self, inp):
        # batch_size * hidden dimension
        z = self.layer(inp)

        # normalize the vectors
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        return norm_z
