import torch
from torch import nn
from torch.nn import functional as F



class NeuralNet(nn.Module):

    def __init__(self, input_size, h1, output_size, activation=nn.ReLU, activate_last=True):
        super(NeuralNet, self).__init__()

        self.activation = activation
        self.linear1 = self._layer(input_size, h1)
        self.linear2 = self._layer(h1, output_size, activate=activate_last)

    def _layer(self, input_size, output_size, batch_norm=True, drop_rate=0.0, activate=True):

        layers = []

        # dropout
        if drop_rate > 0.0:
            layers.append(nn.Dropout(drop_rate))

        # linear/fc layer
        layers.append(nn.Linear(input_size, output_size))

        # batch norm
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_size))

        # activation
        if activate:
            layers.append(self.activation())

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.linear1(x)

        return self.linear2(out)


