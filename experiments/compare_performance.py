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
# @title          :compare_performance.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Compare performance for the developmental network
-------------------------------------------------
Compare the learning speed and final performance of the 
developing network, and a network with fixed architecture.
Note: both networks should have the same architecture in the end.
""" 
import __init__

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pprint
import random
import torch
from torchvision import datasets, transforms

from networks.developmental_network import get_control
from networks.developmental_network import run_network, plot_results
from run_mnist import load_mnist
from utils.spreadsheet_io import write_dict_to_csv

pp = pprint.PrettyPrinter(indent=4)

# Define some parameters
plot = True

def reset_seed(seed=1993):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def eval_network(config, t_dl, v_dl, alpha_0=0., color='k', 
                                                save_title=None, 
                                                figure=None, 
                                                label=None,
                                                init='eye'):
    """Evaluate network, display and print results"""
    #reset_seed()
    print('\nTesting network...')
    print('Initial architecture: %s'%str(config['N']))

    loss = np.zeros((config['epochs'], config['instances']))
    accu = np.zeros((config['epochs'], config['instances']))
    alpha = [[] for i in range(config['instances'])]
    for i in range(config['instances']):
        model, loss[:,i], accu[:,i], alpha[i] = run_network(config, t_dl, v_dl,
                                                v_control=config['v_control'],
                                                h_control=config['h_control'],
                                                alpha_0=alpha_0,
                                                init=init)
    
    config['N'] = model.N
    figure = plot_results(loss, accu, alpha, config, plot_alpha=False,
                                                     save_title=save_title, 
                                                     color=color,
                                                     figure=figure,
                                                     label=label)

    print('Final architecture: %s'%str(model.N))
    return model, loss, accu, config, figure

def get_fixed_params(config):
    """Return modified config dictionary for a fixed (not-growing) network"""
    config_fixed = config.copy()
    config_fixed['grow_h'] = False
    config_fixed['grow_v'] = False
    config_fixed['v_control'] = None
    config_fixed['h_control'] = None
    config_fixed['a_lr'] = 0.
    return config_fixed


# Load network ids
with open("comparisons/network_ids.pkl", "rb") as f:
    networks = pickle.load(f)

# Evaluate non-evaluated network ids
for i, network_id in enumerate(networks['id']):
    if networks['evaluated'][i] == False: # if network hasn't been evaluated
        print('\n************************************************')
        print('Evaluating %s...'%network_id)

        with open("comparisons/configs/%s.pkl"%network_id, "rb") as f:
            config = pickle.load(f)
        print(config['N'])
        N_ini = config['N'].copy()
        config['h_control'] = get_control(config['epochs'], 
                                        grow=config['grow_h'],
                                        gidx=config['h_gidx'],
                                        period=config['h_period'],
                                        offset=config['h_offset'],
                                        end_offset=config['h_end_offset'])
        config['v_control'] = get_control(config['epochs'], 
                                        grow=config['grow_v'],
                                        gidx=config['v_gidx'],
                                        period=config['v_period'],
                                        offset=config['v_offset'],
                                        end_offset=config['v_end_offset'])
        pp.pprint(config)

        # Load MNIST
        if not 't_dl' in globals():
            t_dl, v_dl = load_mnist(config, path='../data')

        # Run developing network
        model_dvp, loss_dvp, accu_dvp, config, fig = eval_network(config, 
                                                        t_dl, v_dl, 
                                                        color='k',
                                                        label='dvp',
                                                        alpha_0=config['a0'])

        # Run fixed large network
        config_fix = get_fixed_params(config)
        model_fix, loss_fix, accu_fix, config_fix, fig= eval_network(config_fix, 
                                                    t_dl, v_dl, 
                                                    color='r',
                                                    label='fix large',
                                                    figure=fig,
                                                    alpha_0=model_dvp.alpha)

        # Run fixed small network
        config_fixs = config_fix.copy()
        config_fixs['N'] = N_ini
        model_fixs, loss_fixs, accu_fixs, config_fixs, fig = eval_network(
                                                    config_fixs, 
                                                    t_dl, v_dl, 
                                                    color='b',
                                                    label='fix small',
                                                    figure=fig)

        fig, ax = fig
        ax[-1].legend()
        fig.savefig('comparisons/plots/' + network_id)

        # Store model characteristics and final results in .csv file
        chars_dict = {'dataset': config['dataset'],
                    'instances': config['instances'],
                       'epochs': config['epochs'],
                            'T': config['T'],
                            'B': config['B'],
                         'w_lr': config['w_lr'],
                        'N_fin': config['N'],
                        'N_ini': N_ini,
                         'a_lr': config['a_lr'],
                      'v_total': np.count_nonzero(config['v_control']),
                      'h_total': np.count_nonzero(config['h_control']),
                       'v_gidx': config['v_gidx'],
                       'h_gidx': config['h_gidx'],
                     'v_period': config['v_period'],
                     'h_period': config['h_period'],
                     'v_offset': config['v_offset'],
                     'h_offset': config['h_offset'],
                     'loss_dvp': loss_dvp[-1][0],
                     'loss_fix': loss_fix[-1][0],
                     'accu_dvp': accu_dvp[-1][0],
                     'accu_fix': accu_fix[-1][0],
                         'uuid': network_id}
        write_dict_to_csv(chars_dict, 'comparisons/results.csv', mode='a')

        # Store training results in pickle file
        results_dict = {'loss_fix': loss_fix,
                        'accu_fix': accu_fix,
                        'loss_dvp': loss_dvp,
                        'accu_dvp': accu_dvp}
        with open("comparisons/results/%s.pkl"%network_id, "wb") as f:
            pickle.dump(results_dict, f)

        # Change evaluation state of the current network configuration
        networks['evaluated'][i] = True
        with open("comparisons/network_ids.pkl", "wb") as f:
            pickle.dump(networks, f)

plt.show()