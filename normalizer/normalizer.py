#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "yqsun@buaa.edu.cn"
__date__ = '2022/04/8 15:07:24'

import pandas as pd
import numpy as np
from yqsun_test.features.tools import auto_get_corelated_group
import matplotlib.pyplot as plt
from utils.utils import df_to_excell


def plt_each_dis(dd, fn=None):
    fig, axes = plt.subplots(4, 9)
    num = 0
    for i in axes:
        for m in i:
            if num < 34:
                m.hist(dd[:, num])
                num += 1
    if fn:
        plt.savefig(fn)
    else:
        plt.show()
    return None


def normalize(data):
    dmin = np.min(data, axis=0)
    dmax = np.max(data, axis=0)
    delta = dmax - dmin
    print(delta)
    scale = 1 / delta
    dd = (data - dmin) * scale
    return dd


def get_normalize_data(feature, target):
    id = feature['id']
    feature = feature.drop('id', axis=1)
    ffe = normalize(feature.values)
    ffe = pd.DataFrame(ffe, columns=feature.columns)
    ffe = ffe.dropna(axis=1, how='any')
    target = target.drop('id', axis=1)
    fd = ffe.join(target)
    fd = fd.set_index(id.values)
    fd.index.name = 'id'
    return fd


def comps_normalized():
    # data = pd.read_excel(r'G:\codes\modnet\yqsun_test\features\opt_features\56_optimal_features.xlsx', header=0)
    # data = data.drop(['Unnamed: 0', 'Hd'], axis=1)

    data = pd.read_excel(r'G:\codes\modnet\yqsun_test\features\raw_features\clean_composition_features.xlsx', header=0)
    data = data.drop(['id'], axis=1)

    target = pd.read_csv(r'G:\codes\modnet\yqsun_test\features\raw_features\comp_targets.csv')['target']

    now_data = get_normalize_data(data, target)
    df_to_excell('clean_comps_feas_target_data_normalized.xlsx', now_data)
    now_data.to_csv('clean_comps_feas_target_data_normalized.csv')

def mab_normalized():
    data = pd.read_excel(r'G:\codes\modnet\yqsun_test\features\raw_features\clean_MAB_features.xlsx', header=0)
    data = data.drop(['id'], axis=1)

    target = pd.read_csv(r'G:\codes\modnet\yqsun_test\features\raw_features\mab_targets.csv')['target']

    # target = pd.read_csv(r'G:\codes\modnet\yqsun_test\features\raw_features\mab_targets.csv')
    now_data = get_normalize_data(data, target)
    df_to_excell('clean_mab_feas_target_data_normalized.xlsx', now_data)
    now_data.to_csv('clean_mab_feas_target_data_normalized.csv')

def normalizer(
        target=r'',
        input=r'',
        clean_csv_name=r''):
    target = pd.read_csv(target)
    # input = pd.read_csv(r'G:\codes\modnet\mab_ml\features\mab_features.csv')
    input = pd.read_csv(input)

    now_data = get_normalize_data(input, target)
    print(now_data)

    df_to_excell(clean_csv_name + '.xlsx', now_data)
    now_data.to_csv(clean_csv_name + '.csv')

    # comps_normalized = now_data.iloc[0: comps_num, :]
    # mabs_normalized = now_data.iloc[comps_num: len(now_data.index), :]
    # if len(comps_normalized.index) + len(mabs_normalized.index) == len(now_data.index):
    #     print('OK')
    #     df_to_excell(r'clean_comps_feas_target_data_normalized.xlsx', comps_normalized)
    #     df_to_excell(r'clean_mabs_feas_target_data_normalized.xlsx', mabs_normalized)

if __name__ == '__main__':
    import os
    # comps_normalized()
    # mab_normalized()
    # normalizer(input=r'G:\codes\modnet\mab_ml\features\site_structure_composition_features.csv')
    # normalizer(input=r'G:\codes\modnet\mab_ml\features\selected_site_structure_composition_features.csv')

    # main_label = 'orth'
    # main_label = 'hex'
    main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    if main_label:
        if main_label == 'hex_orth':
            input_name = 'comp_struc_hex_orth_feas.csv'
        if main_label == 'hex_orth_space':
            input_name = 'comp_space_hex_orth_feas.csv'
        if main_label == 'hex':
            input_name = 'hex_comp_feas.csv'
        if main_label == 'orth':
            input_name = 'orth_comp_feas.csv'

    clean_csv = 'clean_data_normalized'
    target_name = main_label + r'_mab_Ed.csv'
    target = os.path.join(r'G:\codes\modnet\mab_ml\data', target_name)
    input = os.path.join(r'G:\codes\modnet\mab_ml\features', input_name)
    # normalizer(input=r'G:\codes\modnet\mab_ml\features\comp_struc_hex_orth_feas.csv')
    normalizer(target=target, input=input, clean_csv_name = clean_csv)