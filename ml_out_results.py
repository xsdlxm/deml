#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

from utils.utils import write_json, read_json, check_dir_existence
import os

opt_params = {
    'all_5_optimal_features': 5,
    'all_10_optimal_features': 10,
    'all_20_optimal_features': 20,
    'all_30_optimal_features': 30,
    'all_40_optimal_features': 40,
    'all_50_optimal_features': 50,
    'all_60_optimal_features': 60,
    'all_70_optimal_features': 70,
    'all_80_optimal_features': 80,
    'all_90_optimal_features': 90,
    'all_100_optimal_features': 100,
}

hls = [[100, 50], [200, 100], [400, 200], [600, 400], [100, 50, 20], [200, 100, 40], [400, 200, 80],
       [100, 50, 20, 10], [200, 100, 40, 20]]


def check_log(labels, activations, hidden_layers, optimizers, opt_feas_nums):
    train_save_dir = r'train\save_dir'
    test_save_dir = r'test\save_dir'
    results = {}
    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        new_label = opt_feas_num + '_' + label
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

def run_check_log(opt_params, layers_list):
    nums = sorted(list(opt_params.values()))
    results_ = {}
    for hl in layers_list:
        labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(nums, hl)
        results = check_log(labels, activations, hidden_layers, optimizers, opt_feas_nums)
        results_.update(results)
    return results_

def ml_out(opt_params, hls, read_ml_out=False):
    # read_ml_out = True
    if read_ml_out == False:
        results_ = run_check_log(opt_params, hls)
        write_json(results_, os.path.join(r'data', 'data', 'ml_output.json'))
    else:
        results_ = read_json('data/data/ml_output.json')
    return results_

ml_out_results = ml_out(opt_params, hls, False)

### Then plot n_feas_mae_accuracy and n_feas_rmse_accuracy using make_figs.py