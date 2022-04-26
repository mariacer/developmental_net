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
# @title          :all_alpha_experiments_hhist.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Run all alpha-learning experiments
----------------------------------
"""
import __init__

import matplotlib.pyplot as plt 
import numpy as np
import os
import random
import torch
from torch import optim
from torch import nn
import uuid

from experiments.alpha_learning import NN_coupled_neurons
from networks.developmental_network import *
from run_teacher_data import create_layers
from utils.spreadsheet_io import write_dict_to_csv

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)

def sample_data(M, W, b, config, alphas):
    """Sample data from the weights and biases of the teacher
    Inputs: 
        M: number of desired samples
        W: list of weights of the teacher
        b: list of biases of the teacher
        alphas: list of alpha values in layer l+1 of the teacher
    Outputs: 
        x: tensor of dataset inputs  (M x config['N'][0])
        y: tensor of dataset outputs (M x config['N'][-1]"""
    x = torch.randn((M, config['N'][0]), device=config['device'], 
                                         dtype=config['dtype'])
    h = x
    h_list = []
    for l in range(len(config['N']) - 1):
        z = torch.matmul(h, W[l]) + b[l]
        h = nn.functional.leaky_relu(z, negative_slope=alphas[l])
        h_list.append(h)
    y = h
    return h_list[:-1]

config = {'B':100,  # train batch size
          'T':1000, # test batch size
          'M':60000,
          'MV':10000,
          'epochs_coupled':600,
          'epochs_independent':1000,
          'w_lr': 0.01,
          'a_lr': 0., 
          'device':torch.device('cuda:0'),
          'dtype': torch.float,
          'optim_func':optim.SGD,
          'loss_func':nn.MSELoss(),
          'bias':True,
          'plot':True,
          'input_reshape':(lambda x:x),
          'instances':2,
          'mutation_period':5,
          'grow_h':False,
          'grow_v':False}

# Define architecture set
N_set = [[50, 40, 30, 20, 10], [40, 30, 20, 10], [30, 20, 10]]

# Define list of alpha_0 values for the hidden layers of the student
alpha_0_set = [1] 

store_dict = {  'w_lr': config['w_lr'],
                'a_lr': config['a_lr'], 
                'bias': config['bias'],
           'instances': config['instances']}

for NN_model in [NN_coupled_neurons]: #[NN_coupled_neurons, NN]:

    for N in N_set:

        config['N'] = N 

        # Define alpha_teacher_set as a function of number of hidden layers
        if len(config['N']) == 3:
            alpha_teacher_set = [[0], [1], [.5]]
        elif len(config['N']) == 4:
            alpha_teacher_set = [[0, 0], [1, 1], [0, 1], [1, 0]]
        elif len(config['N']) == 5:
            alpha_teacher_set = [[1, 1, 1], [1, 0, 0], [1, 1, 0], 
                                        [0, 0, 1], [0, 1, 1], [0, 0, 0]]

        # Select number of epochs according to the model type
        if NN_model == NN_coupled_neurons:
            config['epochs'] = config['epochs_coupled'] 
        elif NN_model == NN:
            config['epochs'] = config['epochs_independent'] 

        for alpha_teacher in alpha_teacher_set:

            for alpha_0 in alpha_0_set:

                print('Evaluating new model...')
                model_uuid = str(uuid.uuid4())

                alphas = alpha_teacher.copy()
                if alphas == None:
                    alphas = [0. for i in range(len(config['N'])-2)]
                if not(len(alphas) + 2 == len(config['N'])):
                    raise ValueError('The size of alphas vector is not suitable.')
                alphas.append(1) # output linear layer

                W, b = create_layers(config)

                h_list = sample_data(config['M'], W, b, config, alphas)

                plt.figure()
                for layer in range(len(h_list)):
                    plt.subplot(1,len(h_list), layer+1)
                    rand_order = np.random.permutation(np.shape(h_list[0])[0])[:100]
                    plt.hist(h_list[layer][rand_order, :].cpu())
                    plt.title('hidden layer %d'%(layer + 1))
                plt.savefig('alpha_results/hhist/%s.png'%str(alpha_teacher))

plt.show()
#plt.close('all')