import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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

# constants
use_gpu = config.getboolean("training", "use_gpu") and torch.cuda.is_available()

# paths
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

output_path = config.get("testing", "output_path")
mkdir(output_path)

# models
encoder = NeuralNet(
    config.getint("dataset", "patch_size") * config.getint("dataset", "patch_size") * 3,
    100,
    10,
    activation=nn.Tanh
)

# load the state
state_path = config.get("testing", "state_path")
encoder.load_state_dict(torch.load(state_path))

# move to gpu if needed
if use_gpu:
    encoder = encoder.to(torch.device("cuda:0"))

data_path = config.get("testing", "data_path")
temp_path = config.get("testing", "temp_path")

for fn in os.listdir(temp_path):
    print("processing template image")
    path = os.path.join(temp_path, fn)

    # image_output_path = os.path.join(output_path, fn.split(".")[0])
    # mkdir(image_output_path)

    # dataset
    dataset = Patchifier(
        path,
        int(config.get("dataset", "patch_size")),
        whiten=config.getboolean("dataset", "whiten"),
        stride=int(config.get("testing", "stride")),
        sample_rate=config.getfloat("testing", "sample_rate")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("testing", "batch_size")),
        shuffle=config.getboolean("dataset", "shuffle"),
        num_workers=0
    )

    image_activations = torch.FloatTensor([])

    for batch_id, data in enumerate(dataloader):
        current_batch_size = data.size(0)

        if use_gpu:
            data = data.to(torch.device("cuda:0"))

        # push data through model
        with torch.no_grad():
            encoded = encoder(data.view(current_batch_size, -1))

            image_activations = torch.cat([image_activations,
                                           encoded.detach().cpu()], dim=0)
            #print("image_activations", image_activations.size())

    image_activations_np = image_activations.numpy()

    activation_path = "{}/temp_activations.npy".format(temp_path)
    np.save(activation_path, image_activations_np)

    print("saved activations to {}".format(activation_path))

kmeans = KMeans(n_clusters=config.getint("testing", "n_clusters")).fit(image_activations_np)
temp_labels = kmeans.predict(image_activations_np)
temp_label_path = "{}/labels.npy".format(temp_path)
np.save(temp_label_path, temp_labels)
print("finished processing template image")

for num, fn in enumerate(os.listdir(data_path)):
    print("processing {}".format(fn))
    path = os.path.join(data_path, fn)

    # image_output_path = os.path.join(output_path, fn.split(".")[0])
    # mkdir(image_output_path)

    # dataset
    dataset = Patchifier(
        path,
        int(config.get("dataset", "patch_size")),
        whiten=config.getboolean("dataset", "whiten"),
        stride=int(config.get("testing", "stride")),
        sample_rate=config.getfloat("testing", "sample_rate")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("testing", "batch_size")),
        shuffle=config.getboolean("dataset", "shuffle"),
        num_workers=0
    )

    image_activations = torch.FloatTensor([])

    for batch_id, data in enumerate(dataloader):
        current_batch_size = data.size(0)        

        if use_gpu:
            data = data.to(torch.device("cuda:0"))

        # push data through model
        with torch.no_grad():
            encoded = encoder(data.view(current_batch_size, -1))

            image_activations = torch.cat([image_activations,
                                           encoded.detach().cpu()], dim=0)
            #print("image_activations", image_activations.size())

    image_activations_np = image_activations.numpy()

    activation_path = "{}/{}_activations.npy".format(output_path,fn.split(".")[0])
    np.save(activation_path, image_activations_np)
    if num % 500 == 0:
        print("Have saved {} activations".format(num))

    # print("clustering activations (may take some time)")

    labels = kmeans.predict(image_activations_np)
    label_path = "{}/{}_labels.npy".format(output_path,fn.split(".")[0])
    np.save(label_path, labels)
    if num%500 == 0:
        print("Have finished processing {} images".format(num))