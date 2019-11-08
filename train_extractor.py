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

from dataset import TrainDataset, TestDataset
from model import *

if __name__ == '__main__':
    # In[2]:


    CHECKPOINT_FOLDER = 'checkpoints/'
    DATA_FILE = 'E:/xplore_data/data/images.h5'
    DHSGPS_FILE = 'data/dhs_gps.csv'


    # In[3]:


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # In[4]:


    # Size of image that will get fed into neural net
    INPUT_SIZE = 333


    # In[5]:


    dtrain = TrainDataset(h5_file=DATA_FILE, dhsgps_file=DHSGPS_FILE, K=INPUT_SIZE)


    # In[6]:


    def train_model(model, dataloader, optimizer, scheduler, num_epochs=4):
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
                stats = stats.to(device)
                wt = stats[:, 3:14].mean(axis=1)
                wt = wt / wt.sum()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, y) * wt
                    loss = loss.sum()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()


                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data)

            epoch_loss = running_loss / n
            epoch_acc = running_corrects.double() / n

            print('Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
        return model


    # In[7]:


    net = initialize_model2(100)
    net.to(device)
    
#     SAVED_MODEL_PATH = 'checkpoints/mobilenet_6_0'
#     net.load_state_dict(torch.load(SAVED_MODEL_PATH))


    # In[8]:


    BATCH_SIZE = 32
    EPOCHS_PER = 10
    ROUNDS = 5
    
    LR = 0.0001
    WT_DECAY = 0.0001

    # Data loader
    dloader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Create training optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WT_DECAY)
    # Optimizer LR decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    print('LR = %f, wt decay = %f, n_features = %d' % (LR, WT_DECAY, net.n_features))


    # In[ ]:

    for r in range(ROUNDS):
        net = train_model(net, dloader, optimizer, scheduler, EPOCHS_PER)


        # In[ ]:


        checkpoint_name = 'vgg11bn_2_' + str(r)
        torch.save(net.state_dict(), os.path.join(CHECKPOINT_FOLDER, checkpoint_name))


    # In[ ]:




