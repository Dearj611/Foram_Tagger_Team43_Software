# -*- coding: utf-8 -*-
# License: BSD
# Author: Sasank Chilamkurthy
# This model utilizes vgg with the final layer

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch import optim, cuda
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import cv2 as cv
from PIL import Image
import load_data
from load_data import ForamDataSet
from importlib import reload
from timeit import default_timer as timer

plt.ion()   # interactive mode
train_on_gpu = torch.cuda.is_available()
save_file_name = 'vgg16-finetune.pt'
checkpoint_path = 'vgg16-finetune.pth'

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print('Train on gpu: {train_on_gpu}'.format(train_on_gpu=train_on_gpu))
multi_gpu = False
# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print('{gpu_count} gpus detected.'.format(gpu_count=gpu_count))
        

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # val does not use augmentation
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # test does not use augmentation
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # val does not use augmentation
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # test does not use augmentation
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = '../training-images'
image_datasets = {}
image_datasets['train'] = ForamDataSet(csv_file=os.path.join(dirpath, 'train.csv'),
                                               root_dir=data_dir,
                                               master_file='../data-csv/file0.csv',
                                               transform=data_transforms['train'])
image_datasets['val'] = ForamDataSet(csv_file='../data-csv/file{i}.csv'.format(i=order['val']),
                                        root_dir=data_dir,
                                        master_file='../data-csv/file0.csv',
                                        transform=data_transforms['val'])
image_datasets['test'] = ForamDataSet(csv_file='../data-csv/file{i}.csv'.format(i=order['test']),
                                        root_dir=data_dir,
                                        master_file='../data-csv/file0.csv',
                                        transform=data_transforms['test'])                                     
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
classes = image_datasets['train'].labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_better(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          scheduler,
          max_epochs_stop=5,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print('Model has been trained for: {n} epochs.\n'.format(n=model.epochs))
    except:
        model.epochs = 0
        print('Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        scheduler.step()
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
#             print(
#                 'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
#                 end='\r')

        # After training loops ends, start validation
        else:
            print('Epoch: {epoch}, out of: {max_epoch}'.format(epoch=epoch, max_epoch=n_epochs))
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        '\nEpoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}'.format(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss)
                    )
                    print(
                        '\t\tTraining Accuracy: {train_acc}%\t Validation Accuracy: {valid_acc}%'.format(train_acc=(100*train_acc), valid_acc=(100*valid_acc))
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            '\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min} and acc: {valid_acc}%'.format(epoch=epoch, best_epoch=best_epoch, valid_loss_min=valid_loss_min, valid_acc=valid_acc)
                        )
                        total_time = timer() - overall_start
                        print(
                            '{total_time:.2f} total seconds elapsed. {per_epoch} seconds per epoch.'.format(total_time=total_time, per_epoch=total_time/(epoch+1))
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        '\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'.format(best_epoch=best_epoch, valid_loss_min=valid_loss_min, valid_acc=valid_acc)
    )
    print(
        '{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'.format(total_time=total_time, per_epoch=total_time/epoch)
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

def create_model(model_type):
    criteria = {}
    if model_type == 'resnet':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes)) # resets finaly layer
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    if model_type == 'vgg':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():    # unfreeeze the 4th and 5th layers
            if 512 in param.shape:
                break
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, len(classes)), nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        
    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)
    criteria['model'] = model
    criteria['optimizer'] = optimizer
    criteria['criterion'] = criterion
    criteria['train_loader'] = dataloaders['train']
    criteria['valid_loader'] = dataloaders['val']
    criteria['save_file_name'] = save_file_name
    criteria['scheduler'] = exp_lr_scheduler
    criteria['n_epochs'] = 20
    model.idx_to_class = {num:species for num,species in enumerate(image_datasets['train'].labels)}
    model, history = train_better(**criteria)
    return model, history


def accuracy(output, target, topk=(1, )):
    """
    Compute the topk accuracy(s)
    target: the correct labelled answer
    """
    if train_on_gpu:
        output = output.to(device)
        target = target.to(device)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def overall_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        counter = 0
        result = 0
        # Testing loop
        for data, targets in test_loader:
            if train_on_gpu:
                data, targets = data.to(device), targets.to(device)
            # Raw model output
            result += accuracy(model(data), targets)[0]
            counter += 1
    return result/counter
model, history = create_model('resnet')
history.to_csv('history3.csv')
checkpoint = {
    'idx_to_class': model.idx_to_class,
    'epochs': model.epochs,
}

checkpoint['classifier'] = model.classifier
checkpoint['state_dict'] = model.state_dict()

# Add the optimizer
checkpoint['optimizer'] = model.optimizer
checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

acc = overall_accuracy(model, dataloaders['test'])
print('overall accuracy is:', acc)
# Save the data to the path
torch.save(checkpoint, checkpoint_path)