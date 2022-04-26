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
# @title          :run_teacher_data.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Teacher-student framework for studying developmental network
------------------------------------------------------------
- teacher: network with given activation function and architecture
- student: learns from the data of the teacher network
""" 
import numpy as np
import random

from networks.developmental_network import *

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)

def create_layers(config):
    """Create weight and bias tensors for the different layers

    Args:
        config: The config.

    Returns:
        (....): Tuple containing:

        - **W**: list of length of layers-1 with weights for layer l+1 to l.
        - **b**: list of length of layers-1 with biases of layer l+1.
    """
    W = []
    b = []
    for l in range(len(config['N']) - 1):
        W_new = torch.randn((config['N'][l], config['N'][l+1]), 
                                    device=config['device'],  
                                    dtype=config['dtype'])
        nn.init.kaiming_uniform_(W_new, a=np.sqrt(5))
        W.append(W_new)

        b_new = torch.randn(config['N'][l+1], 
                                    device=config['device'], 
                                    dtype=config['dtype'])
        bound = 1./np.sqrt(config['N'][l+1])
        nn.init.uniform_(b_new, -bound, +bound)
        b.append(b_new)
    return (W, b)


def sample_data(M, W, b, config, alphas):
    """Sample data from the weights and biases of the teacher.

    Args: 
        M: number of desired samples
        W: list of weights of the teacher
        b: list of biases of the teacher
        alphas: list of alpha values in layer l+1 of the teacher.

    Returns: 
        x: tensor of dataset inputs  (M x config['N'][0])
        y: tensor of dataset outputs (M x config['N'][-1].
    """
    x = torch.randn((M, config['N'][0]), device=config['device'], 
                                         dtype=config['dtype'])
    h = x
    for l in range(len(config['N']) - 1):
        z = torch.matmul(h, W[l]) + b[l]
        h = nn.functional.leaky_relu(z, negative_slope=alphas[l])
    y = h

    return x, y

def create_dataloader(x, y, B=100):
    """Split x and y tensors into batches of size B and return dataloader.

    Args:
        x: The inputs.
        y: The targets.
        B (int): The batch size.

    Returns:
        out: list (dataloader) with pairs of inputs-output batches.
    """
    out = []
    batches = np.floor(np.shape(x)[0]/B).astype(int)
    for i in range(batches):
        out.append([x[i*B:(i+1)*B,:], y[i*B:(i+1)*B]])

    return out

def generate_data(config, alphas=None):
    """Create training and evaluation dataloaders from a random teacher.

    Args:
        alphas: list of the negative slope of units in each hidden layer.

    Returns: 
        train_dl: training dataloader set
        valid_dl: validation dataloader set.
    """

    # Generate default alphas vector or check input alphas vector
    if alphas == None:
        alphas = [0. for i in range(len(config['N'])-2)]
    if not(len(alphas) + 2 == len(config['N'])):
        raise ValueError('The size of alphas vector is not suitable.')
    alphas.append(1) # output linear layer

    W, b = create_layers(config)

    # Create training set
    x, y = sample_data(config['M'], W, b, config, alphas)
    train_dl = create_dataloader(x, y, B=config['B'])

    # Create validation set
    x, y = sample_data(config['MV'], W, b, config, alphas)
    valid_dl = create_dataloader(x, y, B=config['T'])

    return train_dl, valid_dl 


if __name__=='__main__':

    config = {'N':[30, 20, 10],
              'B':100,    # train batch size
              'T':1000,   # test batch size
              'M':60000,  # samples for training
              'MV':10000, # samples for validation
              'epochs':5, 
              'w_lr':0.05,
              'a_lr': 0., 
              'device':torch.device('cuda:0'),
              'dtype': torch.float,
              'optim_func':optim.SGD,
              'loss_func':nn.MSELoss(),
              'bias':True,
              'plot':False,
              'gidx':1, # layer index for growth
              'mutation_period':2,
              'input_reshape':(lambda x:x),
              'grow_v':False,
              'grow_h':False,
              'h_gidx':1, # layer index for growth
              'v_gidx':1, # layer index for growth
              'instances':1} # frequency to grow

    t_dl, v_dl = generate_data(config)

    loss = np.zeros((config['epochs'], config['instances']))
    alpha = [[] for i in range(config['instances'])]

    for i in range(config['instances']):
      print('\nRunning instance %d/%d'%(i+1, config['instances']))
      _, loss[:,i], _, alpha[i] = run_network(config, t_dl, v_dl)

    plot_results(loss, _, alpha, config)