import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from configuration import config
from dataset import Patchifier
from model import NeuralNet

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from configparser import ConfigParser
config = ConfigParser()
with open("stanosa.conf", "r") as f:
    config.read_file(f)

# KMeansconstants
use_gpu = config.getboolean("training", "use_gpu") and torch.cuda.is_available()

# paths
output_path = config.get("training", "output_path")
state_path = os.path.join(output_path, "states")
for path in [output_path, state_path]:
    try:
        os.makedirs(path)
    except OSError:
        pass

# models
encoder = NeuralNet(
    config.getint("dataset", "patch_size") * config.getint("dataset", "patch_size") * 3,
    100,
    10,
    activation=nn.Tanh
)

decoder = NeuralNet(
    10,
    100,
    config.getint("dataset", "patch_size") * config.getint("dataset", "patch_size") * 3,
    activation=nn.Tanh,
    activate_last=False
)

# move to gpu if needed
if use_gpu:
    encoder = encoder.to(torch.device("cuda:0"))
    decoder = decoder.to(torch.device("cuda:0"))

# optimisers
lr = config.getfloat("training", "lr")
optim_encoder = optim.Adam(encoder.parameters(), lr=lr)
optim_decoder = optim.Adam(decoder.parameters(), lr=lr)

# dataset
dataset = Patchifier(
    config.get("training", "data_path"),
    int(config.get("dataset", "patch_size")),
    whiten=config.getboolean("dataset", "whiten"),
    stride=int(config.get("training", "stride"))
)

dataloader = DataLoader(
    dataset,
    batch_size=int(config.get("training", "batch_size")),
    shuffle=config.getboolean("dataset", "shuffle"),
    num_workers=int(config.get("training", "num_workers"))
)

# loss
criterion = nn.MSELoss()
loss_log = []
batch_count = 0

for epoch in range(config.getint("training", "epochs")):
    for batch_id, data in enumerate(dataloader, 0):
        current_batch_size = data.size(0)        

        if use_gpu:
            data = data.to(torch.device("cuda:0"))

        # clear gradients
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()

        # push data through model
        #encoded = encoder(F.dropout(data.view(current_batch_size, -1), 0.0))
        encoded = encoder(data.view(current_batch_size, -1))
        #print("encoded", encoded.size())
        decoded = decoder(encoded).view(data.size())

        # calculate loss
        loss = criterion(decoded, data)

        # backprop
        loss.backward()

        # update params
        optim_decoder.step()
        optim_encoder.step()

        if batch_id % 500 == 0:
            print("[{}/{}] batch {}, loss={}".format(
                epoch,
                config.getint("training", "epochs"),
                batch_id,
                loss.item()))

            #print("data", data.size(), data.min().item(), data.max().item())                
            #print("decoded", decoded.size(), decoded.min().item(), decoded.max().item())

            loss_log.append([batch_count, loss.item()])

            loss_plot = sns.lineplot(data=pd.DataFrame(loss_log, columns=["batch", "loss"]), x="batch", y="loss")
            #df_loss = df_loss.plot(kind='line', x="batch", y="loss")
            fig = loss_plot.get_figure()
            fig.savefig("{}/loss.png".format(output_path))
            plt.close(fig)

            vutils.save_image(
                data,
                "{}/data.png".format(output_path),
                normalize=True
            )

            vutils.save_image(
                decoded,
                "{}/decoded.png".format(output_path),
                normalize=True
            )

            data_dist = sns.distplot(data.view(-1))
            fig = data_dist.get_figure()
            fig.savefig("{}/data_distribution.png".format(output_path))
            plt.close(fig)

            decoded_dist = sns.distplot(decoded.detach().view(-1))
            fig = decoded_dist.get_figure()
            fig.savefig("{}/decoded_distribution.png".format(output_path))
            plt.close(fig)

        batch_count += 1

    # end of epoch

    # save state
    torch.save(encoder.state_dict(), "{}/encoder_{}.pth".format(state_path,epoch))
    torch.save(decoder.state_dict(), "{}/decoder_{}.pth".format(state_path,epoch))

    # reduce learning rate
    lr *= 0.95
    optim_encoder = optim.Adam(encoder.parameters(), lr=lr)
    optim_decoder = optim.Adam(decoder.parameters(), lr=lr)