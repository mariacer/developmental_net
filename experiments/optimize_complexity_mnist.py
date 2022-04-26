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
# @title          :optimize_complexity_mnist.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Get performance vs. complexity for MNIST
----------------------------------------
""" 
import __init__

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
import pickle
import random
import torch
from torchvision import datasets, transforms

from networks.developmental_network import *
from run_mnist import load_mnist

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)

param_name = 'n_layers' 
suffix = '_' + param_name + ''

if os.path.isfile('optimize_complexity/mnist%s.pkl'%suffix):
    with open("optimize_complexity/mnist%s.pkl"%suffix, "rb") as f:
        results = pickle.load(f)

    loss_set  = results['loss']
    accu_set  = results['accu']
    epochs_mt = results['epochs']
    params_mt = results[param_name]
else:
    config = {'N': [784, 20, 10],
              'B':100,  # train batch size
              'T':1000, # test batch size
              'classify':True,
              'epochs':15, 
              'w_lr':0.05,
              'a_lr': 0., 
              'device':torch.device('cuda:0'),
              'dtype': torch.float,
              'optim_func':optim.SGD,
              'loss_func':F.cross_entropy,
              'grow_v':False,
              'grow_h':True,
              'bias':True,
              'plot':False,
              'h_gidx':1, # layer index for growth
              'v_gidx':1, # layer index for growth
              'instances':3,
              'input_reshape':(lambda x:x.view(-1, 784))} 

    t_dl, v_dl = load_mnist(config, path='../data')

    iter_set = np.array([5, 10, 20, 30, 40, 50, 100, 200, 500, 750, 1000])
    # w_lr --> [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    loss_set = np.zeros((config['epochs'], len(iter_set)))
    accu_set = np.zeros((config['epochs'], len(iter_set)))
    for k, iter_value in enumerate(iter_set):

        config['N'] = [784, iter_value, 10]

        # Train
        loss = np.zeros((config['epochs'], config['instances']))
        accu = np.zeros((config['epochs'], config['instances']))
        alpha = [[] for i in range(config['instances'])]

        for i in range(config['instances']):
          print('\nRunning instance %d/%d'%(i+1, config['instances']))
          model, loss[:,i], accu[:,i], alpha[i] = run_network(config, t_dl, v_dl)

        loss_set[:, k] = np.mean(loss, axis=1)
        accu_set[:, k] = np.mean(accu, axis=1)

    params_mt = np.tile(iter_set, (config['epochs'],1))
    epochs_mt = np.tile(np.arange(config['epochs']), (len(iter_set), 1)).T

    # Store full new list of network configurations
    results = {     'loss': loss_set, 
                    'accu': accu_set,
                  'epochs': epochs_mt,
                param_name: params_mt}
    with open("optimize_complexity/mnist%s.pkl"%suffix, "wb") as f:
        pickle.dump(results, f)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
X, Y, Z = epochs_mt, params_mt, loss_set
surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, linewidth=0)
ax.set_xlabel('epochs')
ax.set_ylabel(param_name)
ax.set_zlabel('loss')

ax = fig.add_subplot(122, projection='3d')
X, Y, Z = epochs_mt, params_mt, accu_set
surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, linewidth=0)
ax.set_xlabel('epochs')
ax.set_ylabel(param_name)
ax.set_zlabel('accuracy %')

fig.savefig('optimize_complexity/mnist%s.png'%suffix)
plt.show()