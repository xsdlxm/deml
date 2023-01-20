#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import torch.nn as nn

def train_for_params(nums=None, hidden_layers=None):
    if nums is None:
        nums = [20]
    else:
        nums = nums

    if hidden_layers is None:
        hidden_layers = [100, 50, 20]
    else:
        hidden_layers = hidden_layers

    for_num = len(nums)
    str_ = '_'.join([str(i) for i in hidden_layers])
    labels = [str_ + "_layer_1000_relu" ] * for_num
    activations = [nn.ReLU()] * for_num
    hidden_layers = [hidden_layers] * for_num
    optimizers = ['Adam'] * for_num
    opt_feas_nums = [str(num) for num in nums]

    sele = lambda x: x

    labels = sele(labels)
    activations = sele(activations)
    hidden_layers = sele(hidden_layers)
    optimizers = sele(optimizers)
    opt_feas_nums = sele(opt_feas_nums)
    return labels, activations, hidden_layers, optimizers, opt_feas_nums

def for_params(nums=[10, 20, 30, 40], hidden_layers=[[600, 400], [400, 200], [100, 50],[200, 100, 40]]):
    print(nums)
    labels = ['_'.join([str(i) for i in hl]) + "_layer_1000_relu" for hl in hidden_layers]
    activations = [nn.ReLU()] * len(nums)
    optimizers = ['Adam'] * len(nums)
    opt_feas_nums = [str(num) for num in nums]

    sele = lambda x: x

    labels = sele(labels)
    activations = sele(activations)
    hidden_layers = sele(hidden_layers)
    optimizers = sele(optimizers)
    opt_feas_nums = sele(opt_feas_nums)
    print(labels, activations, hidden_layers, optimizers, opt_feas_nums)
    return labels, activations, hidden_layers, optimizers, opt_feas_nums

if __name__=="__main__":
    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params()
    print(labels, activations, hidden_layers, optimizers, opt_feas_nums)


