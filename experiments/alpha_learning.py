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
# @title          :alpha_learning.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Teacher-student framework for studying developmental network
------------------------------------------------------------
Student: learns only alphas from hidden layer (input and output are linear)
"""
import __init__

import numpy as np
import os
import random
import torch
from torch import optim
from torch import nn

from networks.developmental_network import *
from run_teacher_data import generate_data

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)

class NN_coupled_neurons(nn.Module):
    """Developmental neural network with layerwise alpha leak terms

    Inputs: 
        config: dictionary with network and simulation parameters
        alpha_0: list of initial alpha values for each hidden layer
        bias: boolean indicating whether to use biases for computations

    Notes:
        lin[l].weight corresponds to connections from layer l to layer l+1
        lin[l].bias corresponds to the bias of layer l+1
        alpha[l] corresponds to the alphas of layer l+1"""
    
    def __init__(self, N, config, alpha_0=0., bias=True):
        super(NN_coupled_neurons, self).__init__()

        self.N  = N
        self.NL = len(N)
        self.bias = bias
        self.name = 'coupled_alpha'

        self.lin = nn.ModuleList()
        self.alpha = nn.ParameterList()
        for k in range(len(config['N'])-1):
            lin_l = nn.Linear(config['N'][k], config['N'][k+1])
            self.lin.append(lin_l.to(config['device']))
            alpha_l = nn.Parameter(torch.tensor(alpha_0, dtype=config['dtype']))
            self.alpha.append(alpha_l) 

    def activation(self, x, l):
        # leaky relu with adjustable layerwise alpha
        x[x<0] *= self.alpha[l]
        return x

    def forward(self, xb):
        x = self.activation(self.lin[0](xb), 0)
        for l in range(1, len(self.lin)):
            x = self.activation(self.lin[l](x), l)
        return x 


if __name__ == '__main__':

    config = {'N':[30, 20, 10],
              'B':100,  # train batch size
              'T':1000, # test batch size
              'M':60000,
              'MV':10000,
              'epochs':5, 
              'w_lr':0.05,
              'a_lr': 0.05, 
              'device':torch.device('cuda:0'),
              'dtype': torch.float,
              'optim_func':optim.SGD,
              'loss_func':nn.MSELoss(),
              'bias':True,
              'plot':True,
              'gidx':1, # layer index for growth
              'mutation_period':2, # frequency to grow
              'input_reshape':(lambda x:x),
              'instances':3} 

    # Chose model: NN or NN_coupled_neurons
    NN_model = NN
    #NN_model = NN_coupled_neurons

    # Define list of alpha_0 values for the hidden layers of the student
    alpha_0_set = [0] #, 1] 

    # Define list alpha values for the hidden layers of the tacher
    alpha_teacher_set = [[0]] #, [1]]

    # Run all params combinations
    for alpha_teacher in alpha_teacher_set:
        for alpha_0 in alpha_0_set:

            print('Evaluating new model...')
            t_dl, v_dl = generate_data(config, alpha_teacher.copy())

            loss  = np.zeros((config['epochs'], config['instances']))
            alpha = [[] for i in range(config['instances'])]

            for i in range(config['instances']):            
                model, loss[:,i], _, alpha[i] = run_network(config, t_dl, v_dl, 
                                                        model=NN_model,
                                                        alpha_0=alpha_0)

            plot_results(loss, _, alpha, config)