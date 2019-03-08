from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from math import log

class ForamDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.master_csv_file = pd.read_csv('../all.csv')
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self.set_labels()

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx, 0])
        species = self.csv_file.iloc[idx, 1]
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels.index(species)

    def set_labels(self):
        labels = list(np.unique(self.master_csv_file['species'].values))
        assert(len(labels) == 17)   # 17 classes for now
        return labels
        
        
if __name__ == "__main__":
    '''
    Data normalization ensures that each input parameter(pixels in this case)
    have a similar data distribution.
    This makes convergence faster while training the network
    Data augmentation just produces perturbed versions of your image, e.g.
    scaling, rotations
    totensor changes (HxWxC) to (CxHxW) in the range [0,1]
    '''
    foramset = ForamDataSet('../train.csv', '../media',
                transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    )
    print(len(foramset))
    sample = foramset[1][0]
    print(sample)
    print(sample.numpy().shape)
    trans = transforms.ToPILImage()
    sample = trans(sample)