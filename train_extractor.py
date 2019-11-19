#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from dataset import TrainDataset, TestDataset
from model import *

# In[1]:

def train_model(model, dataloader, optimizer, scheduler, num_epochs=4, device=None):
    
    n = len(dataloader.dataset)
    for epoch in range(num_epochs):
        print('Epoch {}/{}: '.format(epoch, num_epochs - 1), end='')

        model.train()
        running_loss = 0.0
        running_corrects = 0

        criterion = nn.CrossEntropyLoss(reduction='none')

        # Iterate over data.
        for x, y, stats in dataloader:
            x = x.to(device)
            y = y.to(device)
#             wt = stats[:, 3:14].mean(axis=1)
#             wt = wt / wt.sum()
#             wt = wt.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, y) * wt
                loss = criterion(outputs, y)
                loss = loss.sum()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # statistics
            with torch.no_grad():
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data)
            
            del loss

        epoch_loss = running_loss / n
        epoch_acc = running_corrects.double() / n

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
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
    net.to(device)

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS_PER = 10
    ROUNDS = 5
    
    LR = 1e-4
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
        checkpoint_name = 'vgg11bn_5_' + str(r)
        torch.save(net.state_dict(), os.path.join(CHECKPOINT_FOLDER, checkpoint_name))

    return 0

if __name__ == '__main__':
    main()


