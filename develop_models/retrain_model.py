# This script retrains and re-deploys a new model using the images on Azure Blob Storage

import os
from azure.storage.blob import BlockBlobService
import build_csv
from load_data import ForamDataSet
from model1 import model1
import tempfile
import pandas as pd
import torch
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azureml.core.model import Model
import torch.nn as nn
from torchvision import models


data_dir = './azure-blob-images'    # where all the images will be downloaded to
csv_path = './azure-blobl-csv'
os.mkdir(data_dir)
os.mkdir(csv_path)
container = 'media'
bbs = BlockBlobService(os.environ['AZ_STORAGE_ACCOUNT_NAME'], os.environ['AZ_STORAGE_KEY'])
for blob in bbs.list_blob_names(container):     # download all the images
    bbs.get_blob_to_path(bbs, blob, data_dir)
build_csv.build_files(data_dir, 4, csv_path)

image_datasets = {}
accuracy = []
arrangement =  [[0,1,2,3], [1,2,3,0], [2,3,1,0], [3,0,1,2]]
for num, arr in enumerate(arrangement):
    order = {}
    with tempfile.TemporaryDirectory() as dirpath:
        order['test'] = arr[0]
        order['val'] = arr[1]
        order['train'] = arr[2:]
        if len(order['train']) > 1:
            temp_frame = pd.concat([pd.read_csv('../data-csv/file{num}.csv'.format(num=i))
                                    for i in order['train']])
            temp_frame.to_csv(os.path.join(dirpath, 'train.csv'), encoding='utf-8', index=False)
        image_datasets['train'] = ForamDataSet(csv_file=os.path.join(dirpath, 'train.csv'),
                                               root_dir=data_dir,
                                               master_file='../data-csv/file0.csv',
                                               transform=model1.data_transforms['train'])
        image_datasets['val'] = ForamDataSet(csv_file='../data-csv/file{i}.csv'.format(i=order['val']),
                                             root_dir=data_dir,
                                             master_file='../data-csv/file0.csv',
                                             transform=model1.data_transforms['val'])
        image_datasets['test'] = ForamDataSet(csv_file='../data-csv/file{i}.csv'.format(i=order['test']),
                                              root_dir=data_dir,
                                              master_file='../data-csv/file0.csv',
                                              transform=model1.data_transforms['test'])                                     
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        classes = image_datasets['train'].labels
        model, history = model1.create_model('resnet', classes)    # training starts
        history.to_csv('resnet{i}.csv'.format(i=num))
        checkpoint = {
            'idx_to_class': model.idx_to_class,
        }
        if model1.multi_gpu:
            checkpoint['classifier'] = model.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.fc
            checkpoint['state_dict'] = model.state_dict()
        try:
            checkpoint['optimizer'] = model.optimizer
            checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()
        except Exception as e:
            print(str(e))
        torch.save(checkpoint, 'resnet18-{i}.pth'.format(i=num))
        # torch.save(model.state_dict(), 'resnet18-{i}.pth'.format(i=num))
        print('saved model to' + 'resnet18-{i}.pth'.format(i=num))
        print('accuracy is:', model1.overall_accuracy(model, dataloaders['test']))
        accuracy.append(model1.overall_accuracy(model, dataloaders['test']))


def load_checkpoint(path):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 16)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.idx_to_class = {0: 'G-crassaformis', 1: 'G-elongatus', 2: 'G-hexagonus', 3: 'G-ruber', 4: 'G-sacculifer', 5: 'G-scitula', 6: 'G-siphonifera', 7: 'G-truncatulinoides', 8: 'G-tumida', 9: 'G-ungulata', 10: 'N-acostaensis', 11: 'N-dutertrei', 12: 'N-humerosa', 13: 'O-universa', 14: 'P-obliquiloculata', 15: 'S-dehiscen'}
    return model


# Compare models
best_model = accuracy.index(max(accuracy))
svc_pr = ServicePrincipalAuthentication(
        tenant_id="1faf88fe-a998-4c5b-93c9-210a11d9a5c2",
        service_principal_id="3683e499-d9d1-4b25-9e51-fc0c056415da",
        service_principal_password=os.environ.get("AZUREML_PASSWORD"))
# retrieve the path to the model file using the model name
ws = Workspace.get(name='foram-workspace',
                    subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b',
                    auth=svc_pr)
model_path = Model.get_model_path('resnet18', _workspace=ws)
old_model = load_checkpoint(model_path)
if model1.overall_accuracy(old_model) > accuracy[best_model]:
    pass
else:   # Register new model
    ws = Workspace.get(name='foram-workspace', subscription_id='d90d34f0-1175-4d80-a89e-b74e16c0e31b')
    model = Model.register(model_path='resnet18-{i}.pth'.format(i=best_model),
                           model_name="resnet18",
                           description="inference",
                           workspace=ws)
