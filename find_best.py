#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

from utils.utils import check_dir_existence
from check_log import run_best_target_pred


class best_models():
    def __init__(self, label):
        if label == 'hex_orth':
            # fea_nums = [20, 30],
            # hls = [[200, 100, 40], [400, 200, 80]]
            fea_nums = [10],
            hls = [[100, 50]]
        if label == 'hex_orth_space':
            # fea_nums = [20, 30],
            # hls = [[200, 100, 40], [400, 200, 80]]
            fea_nums = [15],
            hls = [[100, 50]]
        if label == 'hex':
            fea_nums = [10],
            hls = [[400, 200]]
        if label == 'orth':
            fea_nums = [15],
            hls = [[400, 200, 80]]

        self.fea_nums = fea_nums[0]
        # print(self.fea_nums)
        self.hls = hls


if __name__ == "__main__":
    ### According to n_feas_mae_accuracy and n_feas_rmse_accuracy, find best model,
    ### make 'ml_data/DNN/' + new_label + '_tgt_pred.json'
    DNN_dir = r'ml_data\best_DNN'
    check_dir_existence(DNN_dir)

    # main_label = 'orth'
    main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    bests = best_models(main_label)
    fea_nums = bests.fea_nums
    hls = bests.hls
    print(fea_nums)
    print(hls)
    # run_best_target_pred([10, 20, 30], [[400, 200], [200, 100], [400, 200]])  # make ml_data\Ed\MAB\DNN\ml_input.json
    run_best_target_pred(bests.fea_nums, bests.hls, main_label)  # make ml_data\Ed\MAB\DNN\ml_input.json

    ### Then analyze models using analyze_models.py
