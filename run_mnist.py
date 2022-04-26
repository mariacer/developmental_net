#!/usr/bin/env python3
# Copyright 2022 Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :run_mnist.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Run developmental network on MNIST
----------------------------------
"""
import numpy as np
import random
import torch
from torchvision import datasets, transforms

from networks.developmental_network import *

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)


def load_mnist(config, path='data'):
    """Load MNIST dataset and return train and valid dataloaders.

    Args:
        config: The configuration.
        path (str): The path to the data.

    Returns:
        (....): Tuple containing:

        - **t_dl**: The training dataset.
        - **v_dl**: The test dataset.
    """
    B, T = config['B'], config['T']

    # Define transform to normalize the data
    t = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]) # mean/std of MNIST

    # Load train data set and transform, and split in batches
    t_set = datasets.MNIST(root=path, train=True, download=True, transform=t)
    t_dl  = torch.utils.data.DataLoader(t_set, batch_size=B)
          
    v_set = datasets.MNIST(root=path, train=False, download=True, transform=t)
    v_dl  = torch.utils.data.DataLoader(v_set, batch_size=T)

    return t_dl, v_dl


if __name__=='__main__':

    config = {'N':[784, 500, 10],
              'B':100,  # train batch size
              'T':1000, # test batch size
              'classify':True,
              'epochs':5, 
              'w_lr':0.05,
              'a_lr': 0., 
              'device':torch.device('cuda:0'),
              'dtype': torch.float,
              'optim_func':optim.SGD,
              'loss_func':F.cross_entropy,
              'grow_v':False,
              'grow_h':False,
              'bias':True,
              'plot':False,
              'h_gidx':1, # layer index for growth
              'v_gidx':1, # layer index for growth
              'instances':1,
              'input_reshape':(lambda x:x.view(-1, 784))} 

    t_dl, v_dl = load_mnist(config)

    # Create growth controller
    v_control = get_control(config['epochs'], grow=config['grow_v'],
                                              gidx=config['v_gidx'])
    h_control = get_control(config['epochs'], grow=config['grow_h'],
                                              gidx=config['h_gidx'])

    # Train
    loss = np.zeros((config['epochs'], config['instances']))
    accu = np.zeros((config['epochs'], config['instances']))
    alpha = [[] for i in range(config['instances'])]

    for i in range(config['instances']):
      print('\nRunning instance %d/%d'%(i+1, config['instances']))
      model, loss[:,i], accu[:,i], alpha[i] = run_network(config, t_dl, v_dl,
                                                    v_control=v_control,
                                                    h_control=h_control)

    plot_results(loss, accu, alpha, config)


