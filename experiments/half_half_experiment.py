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
# @title          :half_half_experiment.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Initialize neurons in the hidden layer as half linear and half non-linear
-------------------------------------------------------------------------
"""
import __init__

import numpy as np
import random
import torch
from torch import optim
from torch import nn
from tensorboardX import SummaryWriter

from networks.developmental_network import run_network, plot_results

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)


def sample_data(M, W, b, config, alphas):
    """Sample data from the weights and biases of the teacher
    In each layer, the first half of the neurons is nonlinear, and 
    the second half of the neurons is linear"""

    x = torch.randn((M, config['N'][0]), device = config['device'], 
                                         dtype = config['dtype'])
    h = x
    for l in range(len(config['N']) - 1):
        z = torch.matmul(h, W[l]) + b[l]
        h_new = torch.zeros_like(z) 
        h_l   = nn.functional.leaky_relu(z, negative_slope=1.)
        h_nl  = nn.functional.leaky_relu(z, negative_slope=0.)  
        # Trick to have half neurons be linear and half be nonlinear
        middle_idx = int(np.shape(h_new)[1]/2)
        h_new[:, middle_idx:] = h_nl[:, middle_idx:]
        h_new[:, :middle_idx] = h_l[:, :middle_idx]
        h = h_new
    return x, h


if __name__=='__main__':

    # Overwrite sample_data function 
    import run_teacher_data
    run_teacher_data.sample_data = sample_data

    log_dir = 'results/half-half/'

    # Define your parameters
    config = {'N':[30, 20, 10],   # layer dimensions
              'B':100,            # train batch size
              'T':1000,           # test batch size
              'M':60000,          # number of training samples
              'MV':10000,         # number of training samples
              'epochs':500,         # number of epochs
              'w_lr':0.05,        # weight and bias learning rate
              'a_lr': 0.01,       # alphas learning rate
              'device':torch.device('cuda:0'),
              'dtype': torch.float,
              'optim_func':optim.SGD,
              'loss_func':nn.MSELoss(),
              'bias':True,
              'plot':False,
              'input_reshape':(lambda x:x),
              'instances':5} 

    alpha_0_set = [0] #, 1]
    for alpha_0 in alpha_0_set:

        print('Evaluating new model...')
        t_dl, v_dl = run_teacher_data.generate_data(config, [np.nan])

        loss  = np.zeros((config['epochs'], config['instances']))
        alpha = [[] for i in range(config['instances'])]

        for i in range(config['instances']):            
            writer = SummaryWriter('%s'%log_dir)
            writer_title = 'alpha0=%s_i=%d'%(str(alpha_0), i)
            model, loss[:,i], _, alpha[i], =  run_network(config, t_dl, v_dl, 
                                                    alpha_0=alpha_0,
                                                    writer=writer,
                                                    writer_title=writer_title)
            writer.close()

        plot_results(loss, _, alpha, config, save_title=log_dir)

       