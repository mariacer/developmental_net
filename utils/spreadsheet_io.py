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
Code from Paul Brodersen to create spreadsheets with sets of parameters over
which to iterate a simulation
"""
import __init__

from collections import OrderedDict
from copy import deepcopy
import itertools
import numpy as np
import pandas
import uuid

def get_combinations(parameters):
    for values in itertools.product(*list(parameters.values())):
        yield OrderedDict((key, value) for (key, value) in \
            zip(parameters.keys(), values))


def filter_combinations(combinations, *conditions):
    for combination in combinations:
        if len(conditions) > 0:
            if np.all([condition(combination) for condition in conditions]):
                yield combination
        else:
            yield combination

def get_parameter_sets(base_parameter_set, parameters, conditions):
    """Iterate across all possible parameter combinations"""
    for parameter_combination in \
            filter_combinations(get_combinations(parameters), *conditions):
        derived_parameter_set = deepcopy(base_parameter_set)
        derived_parameter_set.update(parameter_combination)
        yield derived_parameter_set

def get_combinations_list(base_parameter_set, parameters):
    """Create a list of parameter sets given the exact combinations"""
    num_combinations = len(list(parameters.values())[0])
    parameter_sets = []
    for i in range(num_combinations):
        current_dict = base_parameter_set.copy()
        for key in parameters.keys():
            current_dict[key] = parameters[key][i]
        parameter_sets.append(current_dict)
    return parameter_sets

def get_uuid():
    return str(uuid.uuid4())


def write_dict_collection_to_csv(dict_collection, file_path, mode='w'):
    for ii, d in enumerate(dict_collection):
        if (ii == 0) and (mode == 'w'):
            write_dict_to_csv(d, file_path, 'w')
        else:
            write_dict_to_csv(d, file_path, 'a')


def write_dict_to_csv(mydict, file_path, mode='w'):
    
    if 'N' in mydict:
        mydict['N'] = str(mydict['N']).replace(',','.')
    if 'alpha_teacher' in mydict:
        mydict['alpha_teacher'] = str(mydict['alpha_teacher']).replace(',','.')
    if 'N_ini' in mydict:
        mydict['N_ini'] = str(mydict['N_ini']).replace(',','.')
    if 'N_fin' in mydict:
        mydict['N_fin'] = str(mydict['N_fin']).replace(',','.')
        
    items = mydict.items()
    keys =   [key   for (key, value) in items]
    values = [value for (key, value) in items]

    # enclose keys in double quotes such that they work well together with inflate
    keys = ['\"{}\"'.format(key) for key in keys]

    # create strings, one per line
    key_string = ','.join(keys)
    value_string = ','.join([str(v) for v in values])

    # terminate strings
    key_string   += '\n'
    value_string += '\n'

    if mode == 'w':
        with open(file_path, mode) as f:
            f.writelines([key_string, value_string])
    elif mode == 'a':
        with open(file_path, mode) as f:
            f.writelines([value_string])


def apply_and_append(filepath_or_dataframe, data_func, func, arguments=None,
    returns=None, filepath_out=None):
    """
    Reads in a spreadsheet / accepts a pandas data frame, loops over
    rows, and interprets columns specified in 'arguments' as arguments
    to a function 'func', and writes out the return values of 'func'
    under the columns in 'returns'.

    The only key word (and thus column header) reserved is 'skip'.
    Rows for which 'skip' evaluates to True are not processed.

    Arguments:
    ----------
        filepath_or_dataframe: str or pandas.DataFrame instance
            File path to a spreadsheet that specifies arguments for func, or
            data frame instance.
        data_func: function handle
            Function from which to laod data of the teacher network whenever a
            new parameter combination is loaded.
        func: function handle
            Function to loop over while passing arguments specified in the
            spreadsheet or data frame.
        arguments: iterable of strings or None (default None)
            Columns corresponding to the arguments of func.
            The column names need to match the argument names of func (the order
            does not matter).
            If None, values from all columns are passed to func.
        returns: iterable of strings or None (default None)
            Columns corresponding to the return values of func.
            If None, function output values are not appended to the output data
            frame.
        filepath_out: str (default None)
            The output file path.
            The output data frame will be saved to that path on processing each
            row. If none, the data frame is not saved, only returned.

    Returns:
    --------
        dataframe: pandas.DataFrame
            The original data frame. Columns specified in `returns`
            are populated wit hthe the output values from func.

    """

    # get data
    if isinstance(filepath_or_dataframe, str):
        df = _read(filepath_or_dataframe)
    elif isinstance(filepath_or_dataframe, pandas.DataFrame):
        df = filepath_or_dataframe
    else:
        raise ValueError("filepath_or_dataframe neither a string nor " + \
                         "a pandas.DataFrame object!" + \
                         "\ntype(filepath_or_dataframe) = {}".\
                         format(type(filepath_or_dataframe)))

    # trim empty rows
    df = df.dropna(how='all')

    if arguments is None:
        arguments = [col for col in df.columns if not col == 'skip']

    kwargs_prev = {}

    # loop over rows
    total_rows = df.shape[0]
    for ii in range(total_rows):

        # test whether to skip row
        if 'skip' in df.columns:
            if df.loc[ii, 'skip']:
                if not np.isnan(df.loc[ii, 'skip']):
                    # nans evaluate to True in python...
                    print("Skipping evaluation of row {}".format(ii))
                    continue # ... to next row without processing current row

        # create dictionary
        kwargs = _row_to_dict(df, ii, arguments)

        # only load data if the set of parameters is different
        if not(kwargs==kwargs_prev):
            data = data_func(**kwargs)

        # pass as key word arguments to function
        return_values = func(data, **kwargs)

        # if the function returns anything, append data to data frame
        if not (returns is None):

            for col, val in zip(returns, return_values):
                df.loc[ii, col] = val

            # save out to file
            if filepath_out:
                _write(df, filepath_out)

        kwargs_prev = kwargs

    return df


def _read(file_path):
    # parse file path to determine extension
    extension = file_path.split('.')[-1]

    # read
    if extension in ('xls', 'xlsx'):
        df = pandas.read_excel(file_path)
    elif extension == 'csv':
        df = pandas.read_csv(file_path)
    else:
        raise ValueError('Spread sheet needs to be a csv or excel ' +
                         '(.xls, .xlsx) file! Extension of supplied ' + 
                         'file path is {}'.format(extension))

    return df


def _write(df, file_path):
    # parse file path to determine extension
    extension = file_path.split('.')[-1]

    # write
    if extension in ('xls', 'xlsx'):
        df.to_excel(file_path, index=False)
    elif extension == 'csv':
        df.to_csv(file_path, index=False)
    else:
        raise NotImplementedError


def _row_to_dict(df, row, keys):
    kv = []
    for key in keys:
        value = df.loc[row, key]
        # check for nan
        if isinstance(value, float): 
            # need to check that type is float as isnan does not accept strings
            if np.isnan(value):
                continue # i.e. skip key, value pair
        value = _parse_value(value)
        kv.append((key, value))

    return dict(kv)

def _parse_value(value):
    if value == 'None':
        value = None
    return value