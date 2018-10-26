import os
from PIL import Image
import numpy as np

import torch 
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from .preprocessing import get_zca_matrix

def calculate_crops(img, image_id, patch_size, stride):
    # define crop windows
    width, height = img.size
    crops = []
    for w in range(0, width-patch_size+1, stride):
        for h in range(0, height-patch_size+1, stride):
            box = (w, h, w + patch_size, h + patch_size)
            crops.append((image_id, box))
    return crops


class Patchifier(Dataset):

    def __init__(self, path_to_images, patch_size, whiten=False, stride=-1, sample_rate=1.0):

        self.path = path_to_images
        self.patch_size = patch_size
        self.stride = patch_size if stride == -1 else stride
        self.sample_rate = sample_rate

        # load in the images
        if os.path.isdir(path_to_images):
            # multiple images
            self.images = []
            for image_id, fn in enumerate(os.listdir(path_to_images)):
                image_path = os.path.join(path_to_images, fn)
                img = Image.open(image_path)
                
                item = {
                    "img": img,
                    "crops": calculate_crops(img, image_id, patch_size, stride)
                }

                tfs = []

                if whiten:
                    zca_matrix = self.zca_matrix(item)
                    tfs.append(transforms.Lambda(lambda x: np.array(x)))
                    tfs.append(transforms.Lambda(lambda x: np.array(x)))
                    tfs.append(transforms.Lambda(lambda x: self.whiten(x, zca_matrix))) # ZCA whitening
                    tfs.append(transforms.Lambda(lambda x: x / (x.std() + 1e-6)))

                tfs.append(transforms.ToTensor())
                item["tf"] = transforms.Compose(tfs)

                self.images.append(item)
        else:
            # single image
            img = Image.open(path_to_images)
            item = {
                "img": img,
                "crops": calculate_crops(img, 0, patch_size, stride)
            }

            tfs = []

            if whiten:
                zca_matrix = self.zca_matrix(item)
                tfs.append(transforms.Lambda(lambda x: np.array(x)))
                tfs.append(transforms.Lambda(lambda x: np.array(x)))
                tfs.append(transforms.Lambda(lambda x: self.whiten(x, zca_matrix))) # ZCA whitening
                tfs.append(transforms.Lambda(lambda x: x / (x.std() + 1e-6)))

            tfs.append(transforms.ToTensor())
            item["tf"] = transforms.Compose(tfs)

            self.images = [item]

        print("self.images", len(self.images))

        all_crops = []
        for item in self.images:
            all_crops = all_crops + item["crops"]
        
        self.all_crops = all_crops
        

    def whiten(self, x, zca_matrix):
        return np.dot(x.reshape(-1), zca_matrix).reshape(x.shape)

    def zca_matrix(self, item):
        
        X = []
        for _, box in item["crops"]:
            patch = np.array(item["img"].crop(box)).reshape(-1)
            X.append(patch)

        X = np.array(X)
        print("X", X.shape)

        if self.sample_rate < 1.0:
            print("sampling X at", self.sample_rate)
            n = int(X.shape[0] * self.sample_rate)
            idx = np.unique(np.random.randint(0, X.shape[0], size=(n,)))
            X = X[idx, ...]

        X = X - X.mean(axis=0) # zero centre
        X = X / (X.std() + 1e-6) # global contrast normalisation

        return get_zca_matrix(X)

    def __len__(self):
        return len(self.all_crops)

    def __getitem__(self, index):
        image_id, box = self.all_crops[index]
        image_item = self.images[image_id]
        patch = image_item["img"].crop(box)
        return image_item["tf"](patch).float()
