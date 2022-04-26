# coding: utf-8
"""
Test suite for developmental_network_mnist

"""

import torch
from torch import optim
from developmental_network import NN, optim_func_partial, train, test
import numpy as np
import torch.nn.functional as F
from functools import partial
from torchvision import datasets, transforms
import random

np.random.seed(1993)
random.seed(1993)
torch.cuda.manual_seed_all(1993)
torch.manual_seed(1993)

batch_size = 100
test_batch_size = 1000

#### Load MNIST dataset
# Define transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]) # mean and std of MNIST images

# Load train data set and transform, and split in batches
train_set = datasets.MNIST(root='../../Data', train=True, download=True, transform=transform)
train_dl  = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
      
valid_set = datasets.MNIST(root='../../Data', train=False, download=True, transform=transform)
valid_dl  = torch.utils.data.DataLoader(valid_set, batch_size=test_batch_size, shuffle=False)

def load_model(path, config, bias):
    model = NN(config['N'], config, bias=bias) 
    model.load_state_dict(torch.load(path))
    model = model.to(config['device'])
    return model

def compare_params(model_1, model_2):
    models_differ = 0
    items_1 = model_1.state_dict().items()
    items_2 = model_2.state_dict().items()
    for key_item_1, key_item_2 in zip(items_1, items_2):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
    else:
        return False

def compare_models(run_function, config, bias=True, params=False):
    """Create model with and without manipulation and compare performance"""

    model = NN(config['N'], config, bias=bias) 
    torch.save(model.state_dict(), path)

    model = load_model(path, config, bias)
    model_0, loss_0   = run_function(model, config, manipulate=False)
    model = load_model(path, config, bias)
    model_1, loss_1 = run_function(model, config, manipulate=True)

    #result = np.array_equal(loss_0, loss_1)
    loss_diff = np.mean(np.abs(np.array(loss_0) - np.array(loss_1)))
    if loss_diff < 5e-7: # there might be some computational inexactitudes
        result = True
    else:
        result = False

    if params:
        # If both params and loss results are equal, return True, else False
        if result and compare_params(model_0, model_1):
            result = True 
        else: 
            result = False
    return result


config = {'N':[784, 20, 10],
          'batch_size':batch_size, 
          'test_batch_size':test_batch_size, 
          'epochs':3, 
          'w_lr':0.1,
          'a_lr': 0., 
          'device':torch.device('cuda:0'),
          'dtype': torch.float,
          'optim_func':optim.SGD,
          'loss_func':F.cross_entropy,
          'input_reshape':(lambda x:x.view(-1, 784))} 

path = 'tests/test_suite_model'

total_failed = 0

################################################################################
#### Check that vertical growth doesn't influence learning if new weights fixed

print('Checking vertical growth function...')

def v_grow(model, config, manipulate=True):
    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==3:
            model.grow_v(config, gidx=gidx, one2one_grad=False)
            opt_function = opt(model, config['w_lr'], config['a_lr']) 
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

result = compare_models(v_grow, config, bias=False)
print('Vertical growth test no bias: %s'%result)
total_failed += not(result)

result = compare_models(v_grow, config, bias=True)
print('Vertical growth test with bias: %s'%result)
total_failed += not(result)

################################################################################
#### Check horizontal growth doesn't influence learning if new weights fixed

print('Checking horizontal growth function...')

# Checking growth and truncate 
def h_grow_and_trunc(model, config, manipulate=True):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==2:
            gidx = 1
            model.grow_h(config, gidx=gidx, 
                                 new_w_grad=False, 
                                 in_w_scale=0., 
                                 out_w_scale=0., 
                                 old_w_scale=1., 
                                 eps=0.)
            model.truncate_horizontally(config)
            opt_function = opt(model, config['w_lr'], config['a_lr']) 
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

result = compare_models(h_grow_and_trunc, config, bias=False, params=True)
print('Vertical growth test no bias: %s'%result)
total_failed += not(result)

result = compare_models(h_grow_and_trunc, config, bias=True, params=True)
print('Vertical growth test with bias: %s'%result)
total_failed += not(result)


# Checking growth, copy and halve weights and freeze them
def run_horizontal_growth(model, config, manipulate=True, bias=False):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==2:
            gidx = 1
            model.grow_h(config, gidx=gidx, 
                                 new_w_grad=False, 
                                 in_w_scale=0.5, 
                                 out_w_scale=1., 
                                 old_w_scale=0.5, 
                                 eps=0.)
            opt_function = opt(model, config['w_lr'], config['a_lr'])
        if epoch== 2: # break right after we are supposed to grow or not
            break
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

#### No bias
bias = False
input_value = next(iter(train_dl))[0].to(config['device'])
model = NN(config['N'], config, bias=bias) 
torch.save(model.state_dict(), path)
model = load_model(path, config, bias)
model_0, loss_0 = run_horizontal_growth(model, config, manipulate=False)
output_original = model_0(input_value).data.detach().cpu().numpy()
model = load_model(path, config, bias)
model_1, loss_1 = run_horizontal_growth(model, config, manipulate=True)
output_manipulate = model_1(input_value).data.detach().cpu().numpy()
output_diff = np.abs(output_original - output_manipulate).mean()
if output_diff < 1e-7: # there might be some computational inexactitudes
    result = True
else:
    result = False
print('Horizontal growth copy and halve weights test no bias: %s'%result)
total_failed += not(result)

#### Bias
bias = True
input_value = next(iter(train_dl))[0].to(config['device'])
model = NN(config['N'], config, bias=bias) 
torch.save(model.state_dict(), path)
model = load_model(path, config, bias)
model_0,   loss_0   = run_horizontal_growth(model, config, manipulate=False)
output_original = model_0(input_value).data.detach().cpu().numpy()
model = load_model(path, config, bias)
model_1, loss_1 = run_horizontal_growth(model, config, manipulate=True)
output_manipulate = model_1(input_value).data.detach().cpu().numpy()
output_diff = np.abs(output_original - output_manipulate).mean()
if output_diff < 1e-7: # there might be some computational inexactitudes
    result = True
else:
    result = False
print('Horizontal growth copy and halve weights test with bias: %s'%result)
total_failed += not(result)


# Checking growth and zero out new weights and freeze weights

def H_grow_and_freeze(model, config, manipulate=True):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==2:
            gidx = 1
            model.grow_h(config, gidx=gidx, 
                                 new_w_grad=False, 
                                 in_w_scale=0., 
                                 out_w_scale=0., 
                                 old_w_scale=1., 
                                 eps=0.)
            opt_function = opt(model, config['w_lr'], config['a_lr']) #
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

result = compare_models(H_grow_and_freeze, config, bias=False)
print('Horizontal growth and zero out new weights test no bias: %s'%result)
total_failed += not(result)

result = compare_models(H_grow_and_freeze, config, bias=True)
print('Horizontal growth and zero out new weights test with bias: %s'%result)
total_failed += not(result)

###############################################################################
#### Check that vertical followed by horizontal growth doesn't influence 
#### learning if new weights fixed

print('Checking vertical followed by horizontal growth functions...')

def VH_grow(model, config, manipulate=True):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==3:
            model.grow_v(config, gidx=gidx, one2one_grad=False)
            model.grow_h(config, gidx=gidx, 
                                 new_w_grad=False,
                                 old_w_scale=1., 
                                 in_w_scale=0., 
                                 out_w_scale=0, eps=0.)
            opt_function = opt(model, config['w_lr'], config['a_lr'])
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

result = compare_models(VH_grow, config, bias=False)
print('Vertical and horizontal growth test no bias: %s'%result)
total_failed += not(result)

result = compare_models(VH_grow, config, bias=True)
print('Vertical and horizontal growth test with bias: %s'%result)
total_failed += not(result)


print('Checking horizontal followed by vertical growth functions...')

def HV_grow(model, config, manipulate=True):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==3: 
            model.grow_h(config, gidx=gidx, 
                                 new_w_grad=False,
                                 old_w_scale=1., 
                                 in_w_scale=0., 
                                 out_w_scale=0, eps=0.)
            model.grow_v(config, gidx=gidx, one2one_grad=False)
            opt_function = opt(model, config['w_lr'], config['a_lr'])
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

result = compare_models(HV_grow, config, bias=False)
print('Horizontal and vertical growth test no bias: %s'%result)
total_failed += not(result)

result = compare_models(HV_grow, config, bias=True)
print('Horizontal and vertical growth test with bias: %s'%result)
total_failed += not(result)


################################################################################
#### Check alpha learning or not

print('Checking alpha learning...')

def run_alphas(model, config, bias=False):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    return model, loss_evolution

bias = True

# don't learn alphas. alphas pre/post should be equal
config['a_lr'] = 0
model = NN(config['N'], config, bias=bias) 
model = model.to(config['device'])
alphas_ini = model.alpha[0].clone()
model_0,   loss_0   = run_alphas(model, config)
alphas_end = model.alpha[0].clone()
result = bool(torch.all(torch.eq(alphas_ini, alphas_end)).item())
total_failed += not(result)

# learn alphas. alphas pre/post should be different
config['a_lr'] = 0.1
model = NN(config['N'], config, bias=bias) 
model = model.to(config['device'])
alphas_ini = model.alpha[0].clone()
model_0,   loss_0   = run_alphas(model, config)
alphas_end = model.alpha[0].clone()
result = not(bool(torch.all(torch.eq(alphas_ini, alphas_end)).item()))

print('Alpha evolution when fixed or learned: %s'%result)
total_failed += not(result)

#### Check alphas are correctly copied during horizontal growth

def run_horizontal_growth(model, config, manipulate=True):

    opt = partial(optim_func_partial, config['optim_func'])
    opt_function = opt(model, config['w_lr'], config['a_lr'])
    gidx = 1
    loss_evolution  = []
    for epoch in range(1, config['epochs'] + 1):
        if manipulate and epoch==2:
            gidx = 1
            alphas_ini = model.alpha[gidx-1].clone()
            model.grow_h(config, gidx=gidx, 
                                            new_w_grad=False)
            opt_function = opt(model, config['w_lr'], config['a_lr']) 
        train(model, train_dl,config, optimizer=opt_function, i_max=10)
        epoch_loss, _ = test(model, valid_dl, config, display=False)
        loss_evolution.append(epoch_loss.item())
    alphas_end = model.alpha[gidx-1][int(model.N[gidx]/2.):].clone()
    alphas_equal = bool(torch.all(torch.eq(alphas_ini, alphas_end)))
    return model, alphas_equal

bias = True
config['N'] = [784, 20, 10]
config['a_lr'] = 0.1
model = NN(config['N'], config, bias=bias) 
model = model.to(config['device'])
model, result = run_horizontal_growth(model, config, manipulate=True)
print('Alphas copying during horizontal growth: %s'%result)
total_failed += not(result)


######################################################################
print('\n%d functions failed the test.'%total_failed)