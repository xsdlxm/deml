#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import os
import pandas as pd
from data.train_label_info import for_params, train_for_params
from data.data_analyzer import read_id_tgt_pred_data, read_test_loss, read_mse_2data, data_linregress
from data.data import hull_out
from utils.utils import write_json, read_json, check_dir_existence


def md_labels(layers_list):
    labels = []
    for hl in layers_list:
        str_ = '_'.join([str(i) for i in hl])
        label = str_
        # label = str_ + "_layer_1000_relu"
        labels.append(label)
    return labels


def read_target_pred(labels, opt_feas_nums, main_label):
    if main_label:
        train_save_dir = os.path.join(r'train', main_label + '_save_dir')
        test_save_dir = os.path.join(r'test', main_label + '_save_dir')
        DNN_dir = os.path.join(r'ml_data', main_label + '_best_DNN')
    else:
        train_save_dir = r'train\save_dir'
        test_save_dir = r'test\save_dir'
        DNN_dir = r'ml_data\best_DNN'

    for i in range(0, len(labels)):
        label = labels[i]
        # activation = activations[i]
        # hidden_layer = hidden_layers[i]
        # optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        new_label = opt_feas_num + '_' + label
        new_opt_num_dir = os.path.join(train_save_dir, new_label)
        train_out = os.path.join(new_opt_num_dir, new_label + '_train.out')
        val_out = os.path.join(new_opt_num_dir, new_label + '_validation.out')
        test_out = os.path.join(test_save_dir, 'results_1000_%s_test.out' % new_label)

        test_data = read_id_tgt_pred_data(test_out)
        train_data = read_id_tgt_pred_data(train_out)
        val_data = read_id_tgt_pred_data(val_out)
        all_data = pd.concat([train_data, val_data, test_data], axis=0, ignore_index=True)
        # val_test_data = pd.concat([val_data, test_data], axis=0, ignore_index=True)

        all_data = all_data.set_index('id')
        print(len(all_data))

        dic_ = all_data.T.to_dict()
        print(len(dic_))

        ml = {}
        for id in dic_.keys():
            ml[id] = dic_[id]['predict']
        tgt_pred_data_path = os.path.join('ml_data', DNN_dir, new_label + '_tgt_pred.json')
        write_json(dic_, tgt_pred_data_path)

        # all_id = all_data['id'].tolist()
        # all_target = all_data['target'].tolist()
        # all_predict = all_data['predict'].tolist()

        # train_target = train_data['target'].tolist()
        # train_predict = train_data['predict'].tolist()
        #
        # val_target = val_data['target'].tolist()
        # val_predict = val_data['predict'].tolist()
        #
        # test_target = test_data['target'].tolist()
        # test_predict = test_data['predict'].tolist()
        #
        # val_test_target = val_test_data['target'].tolist()
        # val_test_predict = val_test_data['predict'].tolist()


def run_best_target_pred(nums, hls, main_label):
    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(nums, hls)
    read_target_pred(labels, opt_feas_nums, main_label)


def find_best(set_key, criterion_key, type, results_):
    to_check = []
    for label in results_.keys():
        test_rmse_accuracy = results_[label][set_key][criterion_key]
        # test_mae_accuracy = results_[label]['test']['mae accuracy']
        # print(test_rmse_accuracy)
        to_check.append(test_rmse_accuracy)
    if type == 'max':
        max_accuracy = max(to_check)
    if type == 'min':
        max_accuracy = min(to_check)
    if type == 'mean':
        max_accuracy = sum(to_check) / len(to_check)

    label = to_check.index(max_accuracy)
    label = list(results_.keys())[label]
    return label


def check_log(labels, activations, hidden_layers, optimizers, opt_feas_nums, main_label):
    if main_label:
        train_save_dir = os.path.join(r'train', main_label + '_save_dir')
        test_save_dir = os.path.join(r'test', main_label + '_save_dir')
    else:
        train_save_dir = r'train\save_dir'
        test_save_dir = r'test\save_dir'
    results = {}
    for i in range(0, len(labels)):
        label = labels[i]
        # activation = activations[i]
        # hidden_layer = hidden_layers[i]
        # optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        new_label = opt_feas_num + '_' + label
        # print(new_label)
        new_opt_num_dir = os.path.join(train_save_dir, new_label)
        train_out = os.path.join(new_opt_num_dir, new_label + '_train.out')
        val_out = os.path.join(new_opt_num_dir, new_label + '_validation.out')
        train_log = os.path.join(new_opt_num_dir, 'running_%s.log' % new_label)
        test_out = os.path.join(test_save_dir, 'results_1000_%s_test.out' % new_label)
        test_log = os.path.join(test_save_dir, '%s_test_loss.log' % new_label)

        test_loss = read_test_loss(test_log)
        test_data = read_id_tgt_pred_data(test_out)
        train_val_mseloss = read_mse_2data(train_log)
        train_data = read_id_tgt_pred_data(train_out)
        val_data = read_id_tgt_pred_data(val_out)
        all_data = pd.concat([train_data, val_data, test_data], axis=0, ignore_index=True)
        val_test_data = pd.concat([val_data, test_data], axis=0, ignore_index=True)

        all_target = all_data['target'].tolist()
        all_predict = all_data['predict'].tolist()

        train_target = train_data['target'].tolist()
        train_predict = train_data['predict'].tolist()

        val_target = val_data['target'].tolist()
        val_predict = val_data['predict'].tolist()
        # print(val_target)
        # print(val_predict)

        """
        15_400_200_layer_1000_relu
[0.1634063, 0.3106906, 0.408668, 0.1905536, 0.170315, 0.4455477, -0.0284755, 0.3381932, 0.1933573, 0.4272806, -0.0355079, 0.3510112, 0.1449229, 0.1039866, 0.4028562, 0.1246251, 0.4654601, 0.1667644, 0.4747303, 0.2538592, 0.2782824, 0.1838014, 0.2573606, 0.2825031, 0.4071005, 0.1851758, 0.4175026, 0.1926357, 0.3096627, 0.2064951, 0.20032, 0.028062, 0.1105719, 0.1007904, 0.1259729, 0.2844537, 0.1809202, 0.083177, 0.0802008, 0.1130647, 0.1803782, 0.0563334, 0.1565862, 0.047884, 0.3181673, 0.2184954]
[0.1842444, 0.2425075, 0.4056156, 0.1685778, 0.2642606, 0.4486682, 0.033617, 0.2878219, 0.1081977, 0.3756094, 0.0718308, 0.2917333, 0.1173025, 0.1804027, 0.4178758, 0.058505, 0.3835948, 0.1846379, 0.5147653, 0.2371992, 0.2703858, 0.1803343, 0.2432501, 0.2928076, 0.3283062, 0.2170519, 0.3769294, 0.2100979, 0.2457892, 0.2252182, 0.1822888, 0.1389584, 0.0983834, 0.1469948, 0.0997185, 0.3050039, 0.2928204, 0.1725584, 0.065843, 0.0641858, 0.1642985, 0.1525888, 0.1620249, 0.0830091, 0.2718376, 0.204]
        """

        test_target = test_data['target'].tolist()
        test_predict = test_data['predict'].tolist()

        val_test_target = val_test_data['target'].tolist()
        val_test_predict = val_test_data['predict'].tolist()

        train_r_value2, train_rmse, train_accuracy, train_mae, train_mae_accuracy = data_linregress(train_target,
                                                                                                    train_predict)
        val_r_value2, val_rmse, val_accuracy, val_mae, val_mae_accuracy = data_linregress(val_target, val_predict)
        test_r_value2, test_rmse, test_accuracy, test_mae, test_mae_accuracy = data_linregress(test_target,
                                                                                               test_predict)
        val_test_r_value2, val_test_rmse, val_test_accuracy, val_test_mae, val_mae_accuracy = data_linregress(
            val_test_target, val_test_predict)
        all_r_value2, all_rmse, all_accuracy, all_mae, all_mae_accuracy = data_linregress(all_target, all_predict)

        results[new_label] = {
            'train': {'r2': train_r_value2, 'rmse': train_rmse, 'rmse accuracy': train_accuracy, 'mae': train_mae,
                      'mae accuracy': train_mae_accuracy, 'train loss': train_val_mseloss['train_loss'].values[-1], },
            'validation': {'r2': val_r_value2, 'rmse': val_rmse, 'rmse accuracy': val_accuracy, 'mae': val_mae,
                           'mae accuracy': val_mae_accuracy, 'val loss': train_val_mseloss['test_loss'].values[-1], },
            'test': {'r2': test_r_value2, 'rmse': test_rmse, 'rmse accuracy': test_accuracy, 'mae': test_mae,
                     'mae accuracy': test_mae_accuracy, 'test loss': test_loss},
            'all': {'r2': all_r_value2, 'rmse': all_rmse, 'rmse accuracy': all_accuracy, 'mae': all_mae,
                    'mae accuracy': all_mae_accuracy, },
            'val_test': {'r2': val_test_r_value2, 'rmse': val_test_rmse, 'rmse accuracy': val_test_accuracy,
                         'mae': val_test_mae, 'mae accuracy': val_mae_accuracy, }
        }
    return results

    # # save_dir = r'G:\codes\modnet\yqsun_test\gjw_ztml_test\save_dir'
    # save_dir = r'G:\codes\modnet\yqsun_test\gjw_ztml_test\new_save_dir'
    # fns = [i for i in os.listdir(save_dir) if '.log' in i]
    # # fn = r'G:\codes\modnet\yqsun_test\gjw_ztml_test\save_dir\running_2layer_100_tanh.log'
    # from pathlib import Path
    #
    # val_results = {}
    # for fn in fns:
    #     fn_path = os.path.join(save_dir, fn)
    #     data = read_mse_2data(fn_path)
    #
    #     aa = pd.DataFrame(data, columns=['epoch', 'step', 'train_loss', 'val_loss'])
    #     last = aa.iloc[-1, -1]
    #     val_results[fn] = last
    #     # print(aa)
    #     # print(last)
    # print(val_results)
    # min_val_loss = min(val_results.values())
    # print( list(val_results.keys())[list(val_results.values()).index(min_val_loss)])
    # print(min_val_loss)


def run_check_log(opt_params, layers_list, main_label):
    nums = sorted(list(opt_params.values()))
    results_ = {}
    for hl in layers_list:
        labels, activations, hidden_layers, optimizers, opt_feas_nums = train_for_params(nums, hl)
        results = check_log(labels, activations, hidden_layers, optimizers, opt_feas_nums, main_label)
        results_.update(results)
    return results_


def ml_out(opt_params, hls, main_label, read_ml_out=False):
    # read_ml_out = True
    if read_ml_out == False:
        # results_ = run_check_log(opt_params, hls, main_label)
        results_ = run_check_log(opt_params, hls, main_label)
        write_json(results_, os.path.join(r'data', 'data', main_label + '_ml_output.json'))
    else:
        results_ = read_json(os.path.join(r'data', 'data', main_label + '_ml_output.json'))
    return results_


if __name__ == '__main__':
    # nums = sorted([i for i in range(20, 220, 20)] + [30, 50, 250] + [i for i in range(300, 1000, 100)])
    opt_params = {
        'all_5_optimal_features': 5,
        'all_10_optimal_features': 10,
        'all_15_optimal_features': 15,
        'all_20_optimal_features': 20,
        'all_25_optimal_features': 25,
        'all_30_optimal_features': 30,
    }

    ### ymax = ymax
    # main_label = 'orth'
    main_label = 'hex'

    ### ymax = ymax -5
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    hls = [[100, 50], [200, 100], [400, 200], [600, 400], [100, 50, 20], [200, 100, 40], [400, 200, 80],
           [100, 50, 20, 10], [200, 100, 40, 20]]
    # hls = [[100, 50], [200, 100]]

    ## Then plot n_feas_mae_accuracy and n_feas_rmse_accuracy using make_figs.py
    ml_out_results = ml_out(opt_params, hls, main_label, read_ml_out=True)

    from plotter.plotter import n_feas_error_plotter

    if main_label:
        train_save_dir = os.path.join(r'train', main_label + '_save_dir')
        test_save_dir = os.path.join(r'test', main_label + '_save_dir')
        figures_dir = main_label + '_figures'
    else:
        figures_dir = 'figures'
        train_save_dir = r'train\save_dir'
        test_save_dir = r'test\save_dir'

    print(train_save_dir)
    print(test_save_dir)
    figures_dir = os.path.join(r'plotter', figures_dir)
    check_dir_existence(figures_dir)

    # n_feas_error_plotter(hls, 'mae', 100, opt_params.values(), 'n_feas_mae_accuracy', main_label, 'test')
    # n_feas_error_plotter(hls, 'mae', 100, [5, 10, 15, 20], 's_n_feas_mae_accuracy', main_label, 'test')
    # # n_feas_error_plotter('rmse', 100, [10, 20, 30, 40], 'n_feas_rmse_accuracy', 'test')
    # n_feas_error_plotter(hls, 'rmse', 100, opt_params.values(), 'n_feas_rmse_accuracy', main_label, 'test')
    # n_feas_error_plotter(hls, 'rmse', 100, [5, 10, 15, 20], 's_n_feas_rmse_accuracy', main_label, 'test')

    n_feas_error_plotter(hls, 'mae', 100, opt_params.values(), 'val_n_feas_mae_accuracy', main_label, 'validation')
    n_feas_error_plotter(hls, 'mae', 100, [5, 10, 15, 20], 'val_s_n_feas_mae_accuracy', main_label, 'validation')
    # n_feas_error_plotter('rmse', 100, [10, 20, 30, 40], 'val_n_feas_rmse_accuracy', 'validation')
    n_feas_error_plotter(hls, 'rmse', 100, opt_params.values(), 'val_n_feas_rmse_accuracy', main_label, 'validation')
    n_feas_error_plotter(hls, 'rmse', 100, [5, 10, 15, 20], 'val_s_n_feas_rmse_accuracy', main_label, 'validation')

    ### Then Run the find_best.py
