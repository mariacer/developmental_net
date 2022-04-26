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
# @title          :all_alpha_experiments.py
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
from run_teacher_data import generate_data
from utils.spreadsheet_io import write_dict_to_csv


np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)


config = {'B':100,  # train batch size
          'T':1000, # test batch size
          'M':60000,
          'MV':10000,
          'epochs_coupled':600,
          'epochs_independent':1000,
          'w_lr': 0.05,
          'a_lr': 0.05, 
          'device':torch.device('cuda:0'),
          'dtype': torch.float,
          'optim_func':optim.SGD,
          'loss_func':nn.MSELoss(),
          'bias':True,
          'plot':True,
          'input_reshape':(lambda x:x),
          'instances':5,
          'mutation_period':5,
          'grow_h':False,
          'grow_v':False}

# Define architecture set
N_set = [[50, 40, 30, 20, 10], [40, 30, 20, 10], [30, 20, 10]]

# Define list of alpha_0 values for the hidden layers of the student
alpha_0_set = [0, 1] 


for NN_model in [NN_coupled_neurons, NN]:
  for N in N_set:
      for w_lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
          config['N'] = N 
          config['w_lr'] = w_lr
          
          store_dict = {  'w_lr': config['w_lr'],
                          'a_lr': config['a_lr'], 
                          'bias': config['bias'],
                     'instances': config['instances']}

          # Define alpha_teacher_set as a function of number of hidden layers
          if len(config['N']) == 3:
              alpha_teacher_set = [[0], [1]]
          elif len(config['N']) == 4:
              alpha_teacher_set = [[0, 0], [0, 1]] #, [1, 0], [1, 1]]
          elif len(config['N']) == 5:
              alpha_teacher_set = [[1, 0, 0], [1, 1, 0]] #
                   #[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 0]]

          # Select number of epochs according to the model type
          if NN_model == NN_coupled_neurons:
              config['epochs'] = config['epochs_coupled'] 
          elif NN_model == NN:
              config['epochs'] = config['epochs_independent'] 

          for alpha_teacher in alpha_teacher_set:

              for alpha_0 in alpha_0_set:

                  print('Evaluating new model...')
                  model_uuid = str(uuid.uuid4())

                  t_dl, v_dl = generate_data(config, alpha_teacher.copy())

                  loss  = np.zeros((config['epochs'], config['instances']))
                  alpha = [[] for i in range(config['instances'])]

                  for i in range(config['instances']):            
                      model, loss[:,i], _, alpha[i] = run_network(config, t_dl, v_dl, 
                                                              model=NN_model,
                                                              alpha_0=alpha_0)

                  title = 'alpha_results/%s'%model_uuid
                  plot_results(loss, _, alpha, config, save_title=title, 
                                                      plot_alpha=True)

                  store_dict['N'] = config['N']
                  store_dict['epochs'] = config['epochs']
                  store_dict['alpha_0'] = alpha_0
                  store_dict['alpha_teacher'] = alpha_teacher
                  store_dict['uuid'] = model_uuid
                  store_dict['model'] = model.name

                  title = 'alpha_results/alpha_learning.csv'
                  write_dict_to_csv(store_dict, title, mode='a')
                  plt.close('all')