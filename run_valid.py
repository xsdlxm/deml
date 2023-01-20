#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import os
from train import ttest
import torch.nn as nn

def use_ml_to_predict_Hd(file_path, fname):
    save_dir = r'G:\codes\modnet\mab_ml\train\save_dir'
    test_dir = r'G:\codes\modnet\mab_ml\test\save_dir'
    hidden_layer = [600, 400]  # [100, 50, 20]  [100, 100, 50, 20]
    label = '600_400_layer_1000_relu'  # '3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'
    activation = nn.ReLU()
    num = 1000

    ttest(file_path, mp_fn=os.path.join(save_dir, 'dnn_params_%d_%s.pkl' % (num, label)),
          test_dir=test_dir, output_fn=os.path.join(test_dir, 'Hd_dnn_params_%d_%s.out' % (num, label)),
          hidden_nodes=hidden_layer, activation=activation, n_output=1,
          batch_normalize=True, dropout=True,
          )


if __name__ == '__main__':
    # head_dir = r'G:\codes\modnet\yqsun_test\features\nomalizer\clean_mabs_feas_target_data_normalized.xlsx'
    head_dir = r'G:\codes\modnet\yqsun_test\features\nomalizer\tolearn\test_set.csv'

    use_ml_to_predict_Hd(head_dir, 'pred')
