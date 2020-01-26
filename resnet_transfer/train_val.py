# -*- coding: utf-8 -
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler

from single_view_train import train_model
from resnet_transfer.config import *

#data_dir = 'data/hymenoptera_data' # has train, val subdirs

# iterate through subdirs and map to transformations, output to dict
#partitions = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                          data_transforms[x])
#                  for x in phases}


# change it to initialize with PyTorch Dataset instance, should spit out image and measurements
dataset = None 
total = len(dataset)


phases = ['train'] # change it to just train if val is no longer needed

#if len(phases) == 2:
#  random_train_indices = np.random.choice(np.arange(total), size=(total//5)*4, replace=False)
#
#  random_val_indices = np.ones(total, dtype=bool)
#  random_val_indices[random_train_indices] = False
# random_val_indices = np.nonzero(random_val_indices)
#
#  split_indices = {'train': random_train_indices, 'val': random_val_indices}
#
#  partitions = {}''train
#  partitions = {x: torch.utils.Subset(dataset, split_indices[x])
#                for x in phases}
#else:
partitions = {'train':dataset}


dataloaders = {x: torch.utils.data.DataLoader(partitions[x], batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
               for x in phases}

partition_sizes = {x: len(partitions[x]) for x in phases}
class_names = partitions['train'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pretrained model
model_pretrained = models.resnet50(pretrained=True, progress=False)
if fix_pretrained_weights:
  for param in model_pretrained.parameters():
      param.requires_grad = False


# replace fc layer and add linear layer
class module_with_concat_linear_layer(nn.Module):
  def __init__(self, pretrained, metadata_dim, output_dim):
    super(module_with_concat_linear_layer, self).__init__()
    self.pretrained = pretrained

    num_ftrs = pretrained.fc.in_features + metadata_dim
    self.fc = nn.Softmax(num_ftrs, num_new_hidden_neurons)
    pretrained.fc = nn.Identity()

    self.linear = nn.modules.Linear(num_new_hidden_neurons, output_dim)
  
  def forward(self, x, metadata):
    x = self.pretrained(x)
    x = torch.cat((x, metadata), 1) # 1 because 0 is for batch dimension
    return self.linear(self.fc(x))

model_ft = module_with_concat_linear_layer(model_pretrained, num_metadata, num_measures)


######################################################################
# Finetuning the convnet
# ----------------------
#
model_ft = model_ft.to(device)

criterion = nn.MSELoss()

if SGD:
  # Observe that all parameters are being optimized
  optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
else:
  optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)
  scheduler = None

model_ft = train_model(dataloaders, model_ft, criterion, optimizer, scheduler,
                       num_epochs=fine_tune_num_epochs)