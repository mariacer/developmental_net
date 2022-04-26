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
# @title          :store_network_config.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/11/2019
# @version        :1.0
# @python_version :3.6.8
"""
Create network configurations to be tested
------------------------------------------
Store the network configurations to be tested in pickle file
"""
import __init__

from collections import OrderedDict
import csv
import dill as pickle
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import optim
import uuid

from utils.spreadsheet_io import get_parameter_sets, get_combinations_list

def delete_folder_content(path):
    filelist = [ f for f in os.listdir(path)]
    for f in filelist:
        os.remove(os.path.join(path, f))

def delete_all_configs():
    """Delete all previous stored results, config files and plots"""
    delete_folder_content('comparisons/configs')
    delete_folder_content('comparisons/plots')
    delete_folder_content('comparisons/results')
    # Reset networks list to empty
    networks = {'id':[], 'evaluated':[]}
    with open("comparisons/network_ids.pkl", "wb") as f:
        pickle.dump(networks, f)
    # Reset results csv to empty
    # with open('comparisons/results.csv', "r") as f:
    #     header = f.readline()
    # with open('comparisons/results.csv', 'w') as f:
    #     f.write(header)

def delete_one_config(name):
    """Delete one configuration"""
    with open("comparisons/network_ids.pkl", "rb") as f:
        networks = pickle.load(f)
        position = networks['id'].index(name)
        _ = networks['id'].pop(position)
        _ = networks['evaluated'].pop(position)
        with open("comparisons/network_ids.pkl", "wb") as f:
            pickle.dump(networks, f)
        os.remove('comparisons/configs/%s.pkl'%name)

def delete_non_evaluated():
    """Delete all non-evaluated models"""
    with open("comparisons/network_ids.pkl", "rb") as f:
        networks = pickle.load(f)
    for i, network_id in enumerate(networks['id']):
        if not networks['evaluated'][i]:
            delete_one_config(network_id)

def display_all_configs():
    with open("comparisons/network_ids.pkl", "rb") as f:
        networks = pickle.load(f)
    for i, network_id in enumerate(networks['id']):
        print('%s evaluated: %s'%(network_id, networks['evaluated'][i]))

#delete_all_configs() 

# Load existing network_ids
with open("comparisons/network_ids.pkl", "rb") as f:
    networks_all = pickle.load(f)

# Define default parameters
base_parameter_set = OrderedDict()      

# Dataset specific
base_parameter_set['N'] = [784, 20, 10]
base_parameter_set['dataset'] = 'mnist'
base_parameter_set['input_reshape']  = (lambda x:x.view(-1, 784))
base_parameter_set['loss_func']  =  F.cross_entropy 
base_parameter_set['classify'] = True

# Growth related   
base_parameter_set['v_control'] = None 
base_parameter_set['h_control'] = None
base_parameter_set['v_period'] = 2
base_parameter_set['h_period'] = 2
base_parameter_set['v_offset'] = 0
base_parameter_set['h_offset'] = 0
base_parameter_set['v_end_offset'] = 0
base_parameter_set['h_end_offset'] = 0
base_parameter_set['v_gidx'] = 1
base_parameter_set['h_gidx'] = 1
base_parameter_set['grow_v'] =  False 
base_parameter_set['grow_h'] =  False

# Training related
base_parameter_set['B'] =  100
base_parameter_set['T'] =  1000  
base_parameter_set['epochs'] =  8 
base_parameter_set['instances'] = 5 
base_parameter_set['w_lr'] =  0.05    
base_parameter_set['a_lr'] =  0.      
base_parameter_set['a0'] =  0.  
base_parameter_set['bias'] =  True  
base_parameter_set['optim_func'] =  optim.SGD

# Miscellaneous
base_parameter_set['plot'] =  True
base_parameter_set['device'] = torch.device('cuda:0')  
base_parameter_set['dtype'] =  torch.float   

###############################################################################
# Define parameters to evaluate

parameters = OrderedDict() 
parameters['v_end_offset'] = [28]
parameters['N'] = [[784, 20, 10]]*len(parameters['v_end_offset'] )
parameters['epochs'] = [30]*len(parameters['v_end_offset'] )
parameters['grow_v'] = [True]*len(parameters['v_end_offset'] )
parameters['v_period'] = [1]*len(parameters['v_end_offset'] )
parameters['v_offset'] = [1]*len(parameters['v_end_offset'] )
parameters['w_lr'] = [0.1]

# from developmental_network import get_control
# import ipdb
# ipdb.set_trace()
# def get_control(epochs, grow=False, gidx=1, period=2, offset=0, end_offset=0)

# only iterate across runs when there is growth in at least 1 dimension
# conditions = [(lambda x: (x['a_lr']!=0. or x['w_lr']!=0.))] 

###############################################################################
# Store all parameters combinations to evaluate

# parameter_sets = get_parameter_sets(base_parameter_set, parameters, conditions)
parameter_sets = get_combinations_list(base_parameter_set, parameters)
parameter_sets = list(parameter_sets)
print('Total number of combinations: %d'%len(parameter_sets))
for parameter_set in parameter_sets:

    # Save each config in an individual pickle file
    network_id = str(uuid.uuid4())
    with open("comparisons/configs/%s.pkl"%network_id, "wb") as f:
        pickle.dump(parameter_set, f)

    networks_all['id'].append(network_id)
    networks_all['evaluated'].append(False)

# Store full new list of network configurations
with open("comparisons/network_ids.pkl", "wb") as f:
    pickle.dump(networks_all, f)
