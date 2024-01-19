import torch
import torch.nn as nn


class DANNDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(DANNDiscriminator, self).__init__()
        self.linear_clf = nn.Sequential()
        self.linear_clf.add_module("fc1", nn.Linear(in_dim, hidden_dim))
        self.linear_clf.add_module("fc1_relu", nn.ReLU(True))
        self.linear_clf.add_module("fc2", nn.Linear(hidden_dim, out_dim))

    def forward(self, input):
        return self.linear_clf(input)
