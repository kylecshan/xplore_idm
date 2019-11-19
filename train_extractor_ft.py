#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from dataset import TrainDataset
from model import *

def train_model(model, dataloader, optimizer, scheduler, num_epochs=4, device=None):
    n = len(dataloader.dataset)
    for epoch in range(num_epochs):
        print('Epoch {}/{}: '.format(epoch, num_epochs - 1), end='')

        model.train()
        running_loss = 0.0
        running_mse = 0

        criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Iterate over data.
        for x, _, stats in dataloader:
            x = x.to(device)
            vaccs = [14, 15, 16, 17, 18, 19, 23]
            y = stats[:, vaccs]
            y = y.to(device)
            wt = stats[:, 3:14].mean(axis=1)
            wt = wt / wt.sum() * x.shape[0]
            wt = wt.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(x)
#                 loss = criterion(outputs, y) * wt
                loss = criterion(outputs, y)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()


            # statistics
            running_loss += loss.item() * x.size(0)
            running_mse += torch.sum((torch.sigmoid(outputs)-y.data)**2)

        epoch_loss = running_loss / n
        epoch_mse = running_mse.double() / n

        print('Loss: {:.4f} MSE: {:.4f}'.format(
            epoch_loss, epoch_mse))
    return model


def main():
    # In[2]:
    CHECKPOINT_FOLDER = 'checkpoints/'
    DATA_FILE = 'E:/xplore_data/data/images.h5'
    DHSGPS_FILE = 'data/dhs_gps.csv'
    INPUT_SIZE = 333

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    dtrain = TrainDataset(h5_file=DATA_FILE, dhsgps_file=DHSGPS_FILE, K=INPUT_SIZE)

    ##########################
    # INITIALIZE+TRAIN MODEL #
    ##########################
    net = initialize_model2()
    vaccs = [14, 15, 16, 17, 18, 19, 23]
    SAVED_MODEL_PATH = 'checkpoints/vgg11bn_4_4'
    net.load_state_dict(torch.load(SAVED_MODEL_PATH))
    net.classifier = nn.Sequential(
        nn.Linear(net.n_features, 50),
        nn.Sigmoid(),
        nn.Linear(50, 20),
        nn.Sigmoid(),
        nn.Linear(20, len(vaccs))
    )
    for i, param in enumerate(net.features.parameters()):
        if i < 28:
            param.requires_grad = False
    
    net.to(device)

    # Training parameters
    BATCH_SIZE = 889
    EPOCHS_PER = 10
    ROUNDS = 5
    
    LR = 1e-6
    WT_DECAY = 1e-4

    # Data loader
    dloader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # Create training optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WT_DECAY)
    # Optimizer LR decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    print('LR = %f, wt decay = %f, n_features = %d' % (LR, WT_DECAY, net.n_features))

    # Train model and save checkpoints
    for r in range(ROUNDS):
        net = train_model(net, dloader, optimizer, scheduler, EPOCHS_PER, device=device)
        checkpoint_name = 'vgg11bn_4_ft_0' + str(r)
        torch.save(net.state_dict(), os.path.join(CHECKPOINT_FOLDER, checkpoint_name))

    return 0

if __name__ == '__main__':
    main()

