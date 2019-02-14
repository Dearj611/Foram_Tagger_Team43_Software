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
import cv2 as cv
from upload.models import Img
from common.load_data import ForamDataSet


def create_csv():
    array = [[os.path.join(os.path.basename(os.path.dirname(str(img.imgLocation))),
              os.path.basename(str(img.imgLocation))), img.species.name]
             for img in Img.objects.all()]
    # array = [[os.path.join(str(img).split('/')[-2:]), img.species.name]
    #          for img in Img.objects.all()]
    df = pd.DataFrame(array)
    df = df.sample(frac=1).reset_index(drop=True)
    df.columns = ['location', 'species']
    boundary_1 = int(len(array)*0.5)
    boundary_2 = int(len(array)*0.75)
    train = df.iloc[:boundary_1]
    val = df.iloc[boundary_1:boundary_2]
    test = df.iloc[boundary_2:]
    # df.to_csv('../all.csv', encoding='utf-8', index=False)
    train.to_csv('../train.csv', encoding='utf-8', index=False)
    val.to_csv('../val.csv', encoding='utf-8', index=False)
    test.to_csv('../test.csv', encoding='utf-8', index=False)
    
    

def get_mean_and_std(root): 
    foramset = ForamDataSet('../train.csv', '../media',
                transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),                
                # transforms.ToTensor()
                ])
    )
    r_mean, g_mean, b_mean = 0, 0, 0
    r_std, g_std, b_std = 0, 0, 0
    total_images = len(foramset)
    for i in range(len(foramset)):
        img = np.asarray(foramset[i][0])/255
        r_mean += img[:,:,0].mean()
        # r_std += np.std(img[:,:,0])
        g_mean += img[:,:,1].mean()
        # g_std += np.std(img[:,:,1])
        b_mean += img[:,:,2].mean()
        # b_std += np.std(img[:,:,2])
    r_mean = r_mean/total_images
    g_mean = g_mean/total_images
    b_mean = b_mean/total_images
    # r_std = r_std/total_images
    # g_std = g_std/total_images
    # b_std = b_std/total_images
    for i in range(len(foramset)):
        img = np.asarray(foramset[i][0])/255
        r_std += np.sum(np.square(img[:,:,0]-r_mean))
        g_std += np.sum(np.square(img[:,:,1]-g_mean))
        b_std += np.sum(np.square(img[:,:,2]-b_mean))
    r_std = np.sqrt(r_std/(224*224*total_images))
    g_std = np.sqrt(g_std/(224*224*total_images))
    b_std = np.sqrt(b_std/(224*224*total_images))
    return ((r_mean,g_mean,b_mean), (r_std,g_std,b_std))


def test_results():
    # mean, std = ((164.46211717195953, 167.2887686830455, 166.27083331850477), (56.65979598187256, 0.0, 0.0))
    # mean, std = ((0.6457506117914041, 0.6568916563439925, 0.6537031968898047), (0.22439587515138493, 0.22821070354822592, 0.2264642688803697))
    mean, std = ((0.6473790889356119, 0.6586621703415373, 0.6545879710236034), (0.2706543851046615, 0.2736628623629193, 0.27078546343785476))
    foramset = ForamDataSet('../train.csv', '../media',
                transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
    )
    print(foramset[0][0])

