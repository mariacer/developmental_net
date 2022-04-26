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
Developmental network
---------------------
Simple NN whose complexity increases by training through:
- increasing the number of layers
- increasing the width of the layers
- increasing the nonlinearities of the neurons
"""
from functools import partial
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class NN(nn.Module):
    """Developmental neural network class

    Args: 
        config: dictionary with network and simulation parameters
        alpha_0: list of initial alpha values for each hidden layer
        bias: boolean indicating whether to use biases for computations

    Notes:
        lin[l].weight corresponds to connections from layer l to layer l+1
        lin[l].bias corresponds to the bias of layer l+1
        alpha[l] corresponds to the alphas of layer l+1"""

    def __init__(self, N, config, alpha_0=0., gidx=1, bias=True):
        super(NN, self).__init__()

        self.N  = N
        self.NL = len(N)
        self.bias = bias
        self.name = 'independent_alpha'
        self.input_reshape = (lambda x: x) # input batch reshape function
        if 'input_reshape' in config:
            self.input_reshape = config['input_reshape']

        # Initialize parameters
        self.init_weights(config)
        self.init_alphas(alpha_0, config)

    def init_weights(self, config):  
        # Initialize weights and biases
        self.lin = nn.ModuleList()
        for l in range(self.NL-1):
            lin_l = nn.Linear(self.N[l], self.N[l+1], bias=self.bias)
            self.lin.append(lin_l.to(config['device']))

    def init_alphas(self, alpha_0, config):
        # Initialize alpha values based on alpha_0

        grad = True
        if config['a_lr'] == 0:
            grad = False

        # Initialize alphas
        self.alpha = nn.ParameterList()
        for l in range(self.NL-2):
            if type(alpha_0)==int or type(alpha_0)==float:
                alpha_l = nn.Parameter(alpha_0*torch.ones((self.N[l+1]), 
                                              dtype=config['dtype'], 
                                              device=config['device']), 
                                        requires_grad=grad)
            else:
                alpha_l = nn.Parameter(alpha_0[l].data, requires_grad=grad)
            self.alpha.append(alpha_l)

        # Make output layer linear
        alpha_l = nn.Parameter(1.*torch.ones((self.N[-1]), 
                                           dtype=config['dtype'],
                                           device=config['device']),
                                    requires_grad=grad)
        self.alpha.append(alpha_l) 

    def activation(self, x, l):
        """ Compute activation of layer l based on input tensor x
        Activation function = leaky relu with independent alpha per neuron"""
        alpha = self.alpha[l].repeat((np.shape(x)[0],1))
        alpha[x>0] = 1
        x = x*alpha
        return x

    def forward(self, xb):
        x = self.input_reshape(xb)
        for l in range(len(self.lin)):
            x = self.activation(self.lin[l](x), l)
        return x 

    def grow_v(self, config, gidx=1, one2one_grad=True, init='eye'):
        """Add linear hidden layers that connect one-to-one to the next layer
        Args: 
            gidx: position of the newly added layer
            one2one_grad: boolean for learning new 1-to-1 connections
            init: type of initialization for new weights
                'eye' for identity mapping
                'random' for random weights"""
        
        # Transform gidx to positive value
        if gidx < 0:
            gidx = self.NL + gidx
        if (gidx<1) or (gidx>=self.NL-1):
            raise ValueError("gidx does not suit network dimensions.")

        # Create weight matrix with square dimensions and add at gidx
        one2one = nn.Linear(self.N[gidx], self.N[gidx], bias=self.bias)
        one2one.to(config['device']) 
        self.lin.insert(gidx, one2one)

        # Create weight connections of value one and zero bias
        if init=='eye':
            self.lin[gidx].weight.data = torch.tensor(np.eye(self.N[gidx]), 
                                                      dtype=config['dtype'], 
                                                      device=config['device'])
        elif init=='random':
            self.lin[gidx].weight.data = torch.tensor(np.random.randn(self.N[gidx], self.N[gidx]), 
                                                      dtype=config['dtype'], 
                                                      device=config['device'])
        if self.bias:
            self.lin[gidx].bias.data  = torch.zeros(self.N[gidx], 
                                                  dtype=config['dtype'], 
                                                  device=config['device']) 

        # Add alphas=1 for the new layer
        old_alpha = self.alpha
        self.alpha = nn.ParameterList()
        shift_by = 0
        for l in range(self.NL):
            if l == gidx-1:
                self.alpha.append(nn.Parameter(torch.ones(self.N[gidx], 
                                                  dtype=config['dtype'],
                                                  device=config['device'])))
                shift_by = 1
            else:
                self.alpha.append(old_alpha[l-shift_by]) 
        
        # Add layer to N vector
        self.N.insert(gidx, self.N[gidx])
        self.NL += 1

        if not(one2one_grad):  # if one2one weights not learned
            self.lin[gidx].weight.requires_grad = False
            if self.bias:
                self.lin[gidx].bias.requires_grad = False 

    def grow_h(self, config, gidx=1, 
                             new_w_grad=True,
                             old_w_scale=0.5,
                             in_w_scale=0.5,
                             out_w_scale=1.,
                             eps=1e-2,
                             init_function=torch.rand_like):
        """Double the size of layer gidx accoring to provided parameters
        Args: 
            gidx: index of the hidden layer to double in size
            new_w_grad: boolean for learning new params (weight, bias and alpha)
            old_w_scale: scaling of the old input weights 
            in_w_scale: scaling of the newly appended input weights
            out_w_scale: scaling of the newly appended output weights
            eps: scaling of initial parameter perturbation
                should be >0 if in_w_scale = out_w_scale = 0, to learn new w
            init_function: function to initialize new weights"""

        # Transform gidx to positive value
        if gidx < 0:
            gidx = self.NL + gidx
        if (gidx<1) or (gidx>=self.NL-1):
            raise ValueError("gidx does not suit network dimensions.")

        # Double incoming weights (and biases if required)
        in_w = self.lin[gidx-1].weight     
        in_grad = in_w.requires_grad   
        if self.bias:
            in_bias = self.lin[gidx-1].bias
        self.lin[gidx-1] = nn.Linear(self.N[gidx-1],self.N[gidx],bias=self.bias)
        self.lin[gidx-1].to(config['device'])
        old_w = old_w_scale*in_w.data
        new_w = in_w_scale*in_w.data + eps*init_function(in_w.data)
        self.lin[gidx-1].weight.data = torch.cat((old_w, new_w))
        self.lin[gidx-1].weight.requires_grad = in_grad
        if self.bias:
            old_b = old_w_scale*in_bias.data
            new_b = in_w_scale*in_bias.data + eps*init_function(in_bias.data)
            self.lin[gidx-1].bias.data = torch.cat((old_b, new_b)) 
            self.lin[gidx-1].bias.requires_grad = in_grad

        # Double outgoing weights (and if required copy biases)
        out_w = self.lin[gidx].weight    
        out_grad = out_w.requires_grad
        if self.bias:
            out_bias = self.lin[gidx].bias
        self.lin[gidx] = nn.Linear(self.N[gidx]*2,self.N[gidx+1],bias=self.bias)
        self.lin[gidx].to(config['device']) 
        new_w = out_w_scale*out_w.data + eps*init_function(out_w.data)
        self.lin[gidx].weight.data = torch.cat((out_w.data, new_w), dim=1)
        self.lin[gidx].weight.requires_grad = out_grad
        if self.bias:
            self.lin[gidx].bias.data = out_bias.data
            self.lin[gidx].bias.requires_grad = out_grad

        # Copy alphas for the new neurons
        old_alpha = self.alpha[gidx-1]
        self.alpha[gidx-1] = nn.Parameter(torch.ones(self.N[gidx]*2, 
                                                    dtype=config['dtype'],
                                                    device=config['device']), 
                                         requires_grad=old_alpha.requires_grad)
        self.alpha[gidx-1].data = torch.cat((old_alpha.data, old_alpha.data))

        # Modify N vector
        self.N[gidx] *= 2
   
        def backward_hook_in(grad): 
            # hook to zero out new incoming weights gradients
            mid_matrix = int(np.shape(grad)[0]/2.)
            out = grad.clone()
            out[mid_matrix:, :] *= 0
            return out

        def backward_hook_out(grad): 
            # hook to zero out new outgoing gradients
            mid_matrix = int(np.shape(grad)[1]/2.)
            out = grad.clone()
            out[:, mid_matrix:] *= 0
            return out

        def backward_hook_vector(grad):
            # hook to zero out mask gradients of a given vector
            mid_matrix = int(len(grad)/2.)
            out = grad.clone()
            out[mid_matrix:] *= 0
            return out

        if not(new_w_grad):  # if no grad for new weights
            if in_grad:
                self.lin[gidx-1].weight.register_hook(backward_hook_in) 
                if self.bias:
                    self.lin[gidx-1].bias.register_hook(backward_hook_vector)
                if old_alpha.requires_grad:
                    self.alpha[gidx-1].register_hook(backward_hook_vector)
            if out_grad:
                self.lin[gidx].weight.register_hook(backward_hook_out)


    def truncate_horizontally(self, config, gidx=1):
        """Divide the width of layer gidx by two
        Args: 
            gidx: position of the layer halve"""
            
        # Transform gidx to positive value
        if gidx < 0:
            gidx = self.NL + gidx
        if (gidx<1) or (gidx>=self.NL-1):
            raise ValueError("gidx does not suit network dimensions.")

        # Halve the number of incoming weights
        new_width = int(self.N[gidx]/2)
        in_w = self.lin[gidx-1].weight
        if self.bias:
            in_bias = self.lin[gidx-1].bias
        self.lin[gidx-1] = nn.Linear(self.N[gidx-1], new_width, bias=self.bias)
        self.lin[gidx-1].to(config['device']) 
        self.lin[gidx-1].weight.data = in_w.data[:new_width, :]
        if self.bias:
            self.lin[gidx-1].bias.data = in_bias.data[:new_width]

        # Halve number of outgoing weights
        out_w = self.lin[gidx].weight
        if self.bias:
            out_bias = self.lin[gidx].bias
        self.lin[gidx] = nn.Linear(new_width, self.N[gidx+1], bias=self.bias)
        self.lin[gidx].to(config['device']) 
        self.lin[gidx].weight.data = out_w.data[:, :new_width]
        if self.bias:
            self.lin[gidx].bias.data = out_bias.data

        # Copy alphas for the new neurons
        old_alpha = self.alpha[gidx-1]
        self.alpha[gidx-1] = nn.Parameter(torch.ones(new_width, 
                                                    dtype=config['dtype'],
                                                    device=config['device']), 
                                         requires_grad=old_alpha.requires_grad)
        self.alpha[gidx-1].data = old_alpha.data[:new_width]

        # Modify N vector
        self.N[gidx] = new_width


def get_control(epochs, grow=False, gidx=1, period=2, offset=0, end_offset=0):
    """Create vector of gidx for v or h growth in each epoch
    Args: 
        grow: boolean indicating whether there will be growth or not
        gidx: manipulation index (+1=first hidden, -2=last hidden)
        period: period for growth operations
        offset: epoch from which the manipulations start
        end_offset: number of epochs to the last to stop manipulations"""
    control = np.zeros(epochs)
    end_offset = epochs - end_offset
    for e in range(epochs):
        if grow and (e%period == 0) and (e>=offset) and (e<=end_offset):
            control[e] = gidx
    return control.astype(int)


def optim_func_partial(optim_func, model, w_lr, a_lr):
    """ Partial optimizer function that only requires params and learning rates
    Args: 
        w_lr: learning rate of weights and biases in all layers
        a_lr:  learning rate of alpha in all hidden layers"""

    params_list = [] 

    # Define learning rates of weights and biases
    for l in range(len(model.lin)):
        params_list.append({'params': model.lin[l].weight,'lr': w_lr})
        if model.bias:
            params_list.append({'params': model.lin[l].bias, 'lr':w_lr})

    # Define learning rates of alphas (0 in output layer)
    for l in range(len(model.lin)-1):
        params_list.append({'params': model.alpha[l], 'lr': a_lr})
    params_list.append({'params': model.alpha[-1], 'lr': 0.}) 

    return optim_func(params_list)


def compute_loss(loss_func, ypred, yb):
    loss = loss_func(ypred, yb)
    return loss


def compute_accuracy(ypred, yb):
    preds = torch.argmax(ypred, dim=1)
    accu = 100*(preds == yb).float().mean().item()
    return accu

def get_weight_norm(model):
    norm = 0
    for l in range(len(model.lin)):
        norm += model.lin[l].weight.data.mean().item()
    return norm/len(model.lin)

def train(model, train_dl, config, optimizer, i_max=np.inf):
    # Train network on train_dl based on a maximum of i_max batches
    model.train()
    for i, (xb, yb) in enumerate(train_dl):
        xb, yb = xb.to(config['device']), yb.to(config['device'])

        # Forward and backward propagate
        optimizer.zero_grad()
        loss = compute_loss(config['loss_func'], model(xb), yb)   
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 2.)
        optimizer.step()
        if i > i_max:
            break

def test(model, valid_dl, config, display=True):
    # Test the network on valid_dl
    model.eval()
    loss = 0
    accu = 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = xb.to(config['device']), yb.to(config['device'])
            ypred = model(xb)
            loss += compute_loss(config['loss_func'], ypred, yb)
            if config['classify']:
                accu += compute_accuracy(ypred, yb)
    loss /= len(valid_dl)
    accu /= len(valid_dl)

    if display and config['classify']:
        print('   test loss: %.3f, test accuracy: %.1f'%(loss, accu))
    elif display:
        print('   test loss: %.3f'%(loss))

    return loss, accu


def list2matrix(l):
    """Convert a list of items with different length into a matrix filled with
    NaNs for values that didn't exist
    Args: 
        l: list of length n_rows and maximum item length of n_cols
    Output:
        l_full: np.array of dimensions n_rows x n_cols """
    n_rows = len(l)
    n_cols = np.max([len(item) for item in l])
    l_full = np.full((n_rows, n_cols), np.nan)
    for i, item in enumerate(l):
        index = n_cols - len(item)
        l_full[i, index:] = item 
    return l_full


def nanfill(alpha_inst, epochs, name):
    """Transform alpha_inst into list of nd.arrays filled with nans
    Args: 
        alpha_inst: list of length number of layers, each item is a list
                         of length epochs, which contains list of alphas
        epochs: total number of epochs in the simulation
        name: name of the model class that was evaluated 
    Outputs: 
        alpha_fill: list of length = number of layers where each item is an
                    np.array of dimension epochs x number of neurons """

    layers = len(alpha_inst)
    alpha_fill = [[] for i in range(layers)]

    for layer in range(layers):
        alpha_layer = alpha_inst[layer]

        # Fill with nans (due to horizontal growth)
        if name=='independent_alpha':
            alpha_layer = list2matrix(alpha_layer)
            layer_neurons = np.shape(alpha_layer)[1]
        elif name=='coupled_alpha':
            alpha_layer = np.array(alpha_layer)[:, np.newaxis]
            layer_neurons = 1

        # Fill with nans (due to vertical growth)
        layer_epochs  = np.shape(alpha_layer)[0]
        if (epochs - layer_epochs) > 0:
            nans = np.full((epochs - layer_epochs, layer_neurons),np.nan)
            alpha_layer = np.concatenate((nans, alpha_layer))

        alpha_fill[layer] = alpha_layer

    return alpha_fill


def plot_performance(loss, accu, config,  figure=None, color='k', 
                                                        label=None,  
                                                        save_title=None):
    """ Plot the evolution of the loss and accuracy
    Args: 
        loss: np.array of dimensions (epochs x instances)
        accu: np.array of dimensions (epochs x instances)
        figure: tuple of (fig, ax) handlers where to make the plot
        color: color to use in the plots
        label: label to use for the legend
        save_title: name under which the figures will be saved"""

    batch_size = config['B']    
    figsize = (6, 4)
    n_subplots = 1
    if config['classify']:
        figsize = (12, 4)
        n_subplots = 2

    if figure == None:
        fig, ax = plt.subplots(1, n_subplots, figsize=figsize)
        if not config['classify']:
            ax = [ax]
    else:
        fig, ax = figure

    num_epochs = np.shape(accu)[0]
    x_vector = np.arange(0, num_epochs).astype(int)

    # Plot loss
    y_vector = np.mean(loss, axis=1)
    y_std = np.std(loss, axis=1)
    ax[0].plot(x_vector, y_vector, color, label=label)
    ax[0].fill_between(x_vector, y_vector-y_std, y_vector+y_std, alpha=0.2,   
                                                               color=color)
    ax[0].set_xlabel('n epochs')
    ax[0].set_ylabel('test loss')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot accuracy
    if config['classify']:
        y_vector = np.mean(accu, axis=1)
        y_std = np.std(accu, axis=1)
        ax[1].plot(x_vector, y_vector, color=color, label=label)
        ax[1].fill_between(x_vector, y_vector-y_std, y_vector+y_std, alpha=0.2,
                                                                   color=color)
        ax[1].set_xlabel('n epochs')
        ax[1].set_ylabel('% test accuracy')
        #ax[1].set_ylim(0, 100)
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_title:
        fig.savefig(save_title + '_perf.png')

    return (fig, ax)


def adapt_for_plot(alpha):
    """Transform alpha to list with first dimension = layers
    Args:
        alpha: list with dim instances > layers > epochs > neurons
    Output:
        alpha_reshape: list with dim layers > epochs > instances > neurons"""
    instances = len(alpha)
    layers = len(alpha[0])
    epochs = np.max([len(i) for i in alpha[0]])
    alpha_reshape = [[] for l in range(layers)]
    for l in range(layers):
        neurons = alpha[0][l].shape[1]
        values = np.zeros((epochs, neurons, instances))
        for i in range(instances):
            values[:,:,i] = alpha[i][l]
        alpha_reshape[l] = np.transpose(values, (0, 2, 1))
    return alpha_reshape


def plot_alphas(alpha, batch_size, config, save_title=None, **kwargs):
    """ Plot the evolution of the alphas
    Args:
        save_title: name under which the figures will be saved"""

    # Reshape alpha to have as first dimension the number of layers
    alpha = adapt_for_plot(alpha)

    if (len(alpha) != len(config['N'])-1) or \
                (len(alpha[0]) != config['epochs']) or \
                (len(alpha[0][0]) != config['instances']):
        raise ValueError('Error in dimensions of alphas to plot.')

    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    num_epochs = np.max([len(i) for i in alpha])
    num_layers = len(alpha)
    x_vec = np.arange(0, num_epochs).astype(int)

    for l in range(num_layers-1): 

        # For independent alphas in each layer
        if isinstance(alpha[l][0][0], np.ndarray): 
            y_vec = np.zeros(num_epochs)
            y_std = np.zeros(num_epochs)
            for epoch in range(num_epochs):
                all_alphas = [y for x in alpha[l][epoch] for y in x]
                y_vec[epoch] = np.mean(all_alphas)
                y_std[epoch] = np.std(all_alphas)

        # For coupled alphas in each layer
        else:  
            y_vec = np.nanmean(alpha[l], axis=1).tolist()
            y_std = np.nanstd( alpha[l], axis=1).tolist()
            y_vec = np.array([np.nan]*(len(x_vec)-len(y_vec)) + y_vec) 
            y_std = np.array([np.nan]*(len(x_vec)-len(y_std)) + y_std)
        ax.plot(x_vec, y_vec)
        ax.fill_between(x_vec, y_vec - y_std, y_vec + y_std, alpha=0.2)

    ax.legend(['hidden layer %i'%i for i in range(1,num_layers)])
    ax.set_xlabel('n epochs')
    ax.set_ylabel('alphas distribution')
    ax.set_ylim(-0.5,1.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_title:
        fig.savefig(save_title + '_alpha.png')

    return (fig, ax)


def plot_results(loss, accu, alpha, config, plot_alpha=True, 
                                            show=False,
                                            **kwargs):
    """Plot evolution of the loss and of the alpha values
    Args:
        plot_alpha: boolean indicating whether to make alpha plots
        show: boolean indicating whether to show the figures or not"""

    if config['plot']:
        figure = plot_performance(loss, accu, config, **kwargs) 
        if plot_alpha:
            _ = plot_alphas(alpha, config['B'], config, **kwargs)
    else: 
        figure = None
    if show:
        plt.show()

    return figure

def run_network(config, train_dl, valid_dl, model=NN,
                                            v_control=None,
                                            h_control=None,
                                            writer=False,
                                            writer_title='',
                                            init='eye', 
                                            **kwargs):
    """Train network given config params, and train and valid dataloaders
    Args:
        model: defaults to NN with independent alphas for each neuron
        v_control: vector of length epochs with gidx for v growth
        h_control: vector of length epochs with gidx for v growth
        writer: receives tensorboard writer, or defaults to fault
        writer_title: save title of tensorboard writer
        kwargs: optional extra arguments for instantiating the model"""

    # If no gidx vectors, create default to 0 -> no growth at any epoch
    if not type(v_control) == np.ndarray:
        v_control = np.zeros(config['epochs'])
    if not type(h_control) == np.ndarray:
        h_control = np.zeros(config['epochs'])
    if 'classify' not in config:
        config['classify'] = False

    #### Evaluate networks
    N = config['N'].copy()
    model = model(N, config, bias=config['bias'], **kwargs) 
    model = model.to(config['device'])

    # Create optimization function
    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])

    # Create variables to keep track of learning
    loss_inst  = np.zeros(config['epochs'])
    accu_inst  = np.zeros(config['epochs'])
    alpha_inst = [[] for i in range(len(model.alpha))]

    # Iterate over training set
    for e in range(0, config['epochs']):

        if v_control[e]:
            print('     -> adding one extra layer in %.f'%v_control[e])
            model.grow_v(config, gidx=v_control[e], init=init)
            opt_function = opt(model, config['w_lr'], config['a_lr'])
            alpha_inst.insert(v_control[e]-1, [])

        if h_control[e]:
            print('     -> doubling the width of layer %.f'%h_control[e])
            model.grow_h(config, gidx=h_control[e])
            opt_function = opt(model, config['w_lr'], config['a_lr'])

        # # plot weights
        # plt.figure()
        # layers = len(model.lin)
        # weights = model.lin[1].weight.data.cpu().numpy()
        # plt.imshow(weights)
        # plt.colorbar()
        # print(e, np.mean(weights), np.std(weights))

        # Train and test
        train(model, train_dl, config, optimizer=opt_function)
        loss_e, accu_e = test(model, valid_dl, config)

        if writer:
            writer_data = model.alpha[0].cpu().data.numpy()
            writer.add_histogram(writer_title, writer_data, e)

        # Keep track of the evolution
        loss_inst[e] = loss_e
        accu_inst[e] = accu_e
        for l in range(len(model.alpha)):
            alpha_inst[l].append(model.alpha[l].cpu().data.numpy())

    alpha_inst = nanfill(alpha_inst, config['epochs'], name=model.name)

    return model, loss_inst, accu_inst, alpha_inst
