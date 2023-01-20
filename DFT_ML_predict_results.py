#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

from data.data_analyzer import read_id_tgt_pred_data, read_test_loss, read_mse_2data
from data.train_label_info import for_params
import os, pprint
import pandas as pd
from utils.utils import check_dir_existence, read_json, write_json
from find_best import best_models


def set_predict_json(set_id_json, pred_json, id_json, set_name, main_label):
    for id in pred_json:
        if id in id_json.keys():
            set_id_json[id] = id_json[id]
            set_id_json[id]['predict'] = pred_json[id]['predict']
            set_id_json[id]['set'] = set_name
            set_id_json[id]['main_label'] = main_label

    return set_id_json


def read_DFT_hull(id_json_list=None, id_json_path=None):
    if id_json_list:
        id_json = {}
        for j in id_json_list:
            js = read_json(j)
            id_json.update(js)
    else:
        id_json = read_json(id_json_path)
    return id_json


def set_stability(set_id_json, thresh_range):
    for k in set_id_json:
        _dict = set_id_json[k]
        Ed = _dict["hull_distance"]
        # Ef = _dict["formation_energy"]
        pred_Ed = _dict["predict"]
        if Ed <= thresh_range[0]:
            set_id_json[k]["stability"] = 'stable'
        if thresh_range[0] <= Ed < thresh_range[1]:
            set_id_json[k]["stability"] = 'metastable'
        if Ed >= thresh_range[1]:
            set_id_json[k]["stability"] = 'unstable'
        if pred_Ed <= thresh_range[0]:
            set_id_json[k]["pred_stability"] = 'stable'
        if thresh_range[0] < pred_Ed <= thresh_range[1]:
            set_id_json[k]["pred_stability"] = 'metastable'
        if pred_Ed > thresh_range[1]:
            set_id_json[k]["pred_stability"] = 'unstable'
    return set_id_json

if __name__ == '__main__':
    is_written = True
    # is_written = False

    # main_label = 'orth'
    main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    if main_label:
        main_label_dir = main_label + '_save_dir'

        if main_label == 'hex':
            id_json = read_DFT_hull(id_json_path='data\data\id_CV_outputs_hex.json')
            space_group = 'hex'

        if main_label == 'orth':
            id_json = read_DFT_hull(id_json_path='data\data\id_CV_outputs_orth.json')
            space_group = 'orth'

        if main_label == 'hex_orth_space':
            id_json = read_DFT_hull(
                id_json_list=['data\data\id_CV_outputs_orth.json',
                              'data\data\id_CV_outputs_hex.json'])
            space_group = 'hex_orth'
        if main_label == 'hex_orth':
            id_json = read_DFT_hull(
                id_json_list=['data\data\id_CV_outputs_orth.json',
                              'data\data\id_CV_outputs_hex.json'])
            space_group = 'hex_orth'
    else:
        main_label_dir = 'save_dir'
    print(space_group)
    # pprint.pprint(id_json)

    train_save_dir = os.path.join(r'train', main_label_dir)
    test_save_dir = os.path.join(r'test', main_label_dir)

    check_dir_existence(train_save_dir)
    check_dir_existence(test_save_dir)

    best = best_models(main_label)
    hls = best.hls
    fea_nums = best.fea_nums
    # print(hls)
    # print(fea_nums)

    ### According to the results of n_feas_mae_accuracy.jpg and n_feas_rmse_accuracy.jpg
    # # labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params([10, 20, 30, 40], [[600, 400], [400, 200], [100, 50],[200, 100, 40]])
    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(fea_nums, hls)

    results = {}
    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        module_label = opt_feas_num + '_' + label
        new_opt_num_dir = os.path.join(train_save_dir, module_label)
        train_out = os.path.join(new_opt_num_dir, module_label + '_train.out')
        val_out = os.path.join(new_opt_num_dir, module_label + '_validation.out')
        train_log = os.path.join(new_opt_num_dir, 'running_%s.log' % module_label)
        test_out = os.path.join(test_save_dir, 'results_1000_%s_test.out' % module_label)
        test_log = os.path.join(test_save_dir, '%s_test_loss.log' % module_label)

        test_loss = read_test_loss(test_log)
        test_data = read_id_tgt_pred_data(test_out).set_index('id').T
        train_val_mseloss = read_mse_2data(train_log)

        train_data = read_id_tgt_pred_data(train_out).set_index('id').T
        val_data = read_id_tgt_pred_data(val_out).set_index('id').T

        test_json = test_data.to_dict()
        train_json = train_data.to_dict()
        val_json = val_data.to_dict()
        pprint.pprint(test_json)

        set_id_json = {}
        set_id_json = set_predict_json(set_id_json, test_json, id_json, 'test', main_label)
        set_id_json = set_predict_json(set_id_json, val_json, id_json, 'validation', main_label)
        set_id_json = set_predict_json(set_id_json, train_json, id_json, 'train', main_label)
        pprint.pprint(set_id_json)
        set_id_json = set_stability(set_id_json, [0.00, 0.07])
        pprint.pprint(set_id_json)
        # def set_formula_key
        # pprint.pprint(set_id_json)

        if is_written == True:
            write_json(set_id_json, 'data/data/%s_%s_set_id.json' % (main_label, module_label))
            # write_json(set_id_json, 'data/data/%s_%s_set_id.json' % (main_label, module_label))

        test_stable = []
        test_pred_stable = []
        test_both_stable = []
        for id in set_id_json:
            _dict = set_id_json[id]
            stability = _dict['stability']
            pred_stability = _dict['pred_stability']
            formula = _dict['formula']
            set_name = _dict['set']
            if set_name == 'test':
                if pred_stability != 'unstable':
                    test_pred_stable.append(_dict)
                if stability != 'unstable':
                    test_stable.append(_dict)
                if pred_stability != 'unstable' and stability != 'unstable':
                    test_both_stable.append(_dict)

        test_both_stable_formula = [[dic['formula'], dic['space_group']] for dic in test_both_stable]
        test_pred_stable_formula = [[dic['formula'], dic['space_group']] for dic in test_pred_stable]
        test_stable_formula = [[dic['formula'], dic['space_group']] for dic in test_stable]
        pprint.pprint(test_both_stable_formula)
        pprint.pprint(test_pred_stable_formula)
        pprint.pprint(test_stable_formula)
        """
        [['Ti2Al1B2', 'orth']]
        [['Al1V2B2', 'orth'], ['Al1B2Ru2', 'orth'], ['Ti2Al1B2', 'orth']]
        
        [['Hf2Sn1B2', 'hex'],
         ['Zr2Al1B2', 'hex'],
         ['Hf2B2Pb1', 'hex'],
         ['Si1B2Mo2', 'hex'],
         ['Zr2Cd1B2', 'hex'],
         ['Hf2Ga1B2', 'hex']]
        [['Hf2Sn1B2', 'hex'],
         ['Zr2Al1B2', 'hex'],
         ['Hf2B2Pb1', 'hex'],
         ['Nb2B2S1', 'hex'],
         ['Si1B2Mo2', 'hex'],
         ['V2Ge1B2', 'hex'],
         ['Zr2Cd1B2', 'hex'],
         ['Hf2B2P1', 'hex'],
         ['Y2B2Pb1', 'hex'],
         ['Hf2Ga1B2', 'hex'],
         ['Hf2Zn1B2', 'hex'],
         ['Nb2Sn1B2', 'hex'],
         ['Y2Tl1B2', 'hex'],
         ['Zr2B2P1', 'hex']]
        
        """
        # pprint.pprint(test_both_stable)
        # pprint.pprint(test_stable)
        # pprint.pprint(test_pred_stable)
        print(len(test_both_stable))
        print(len(test_stable))
        print(len(test_pred_stable))

        # try:
        #     test_TP = len(test_both_stable)
        #     test_TP_FN = len(test_stable)
        #     test_TP_FP = len(test_pred_stable)
        #     recall = test_TP/test_TP_FN
        #     precision = test_TP/test_TP_FP
        #     print(recall)
        #     print(precision)
        # except ZeroDivisionError as e:
        #     print(e)

        """
        [['Zr2Sn1B2', 'hex'],
         ['Hf2In1B2', 'hex'],
         ['Zr2In1B2', 'hex'],
         ['Al1Tc2B2', 'orth'],
         ['Zr2Tl1B2', 'hex'],
         ['Ga1Tc2B2', 'orth'],
         ['Ga1B2Mo2', 'hex'],
         ['Ti2In1B2', 'hex'],
         ['Hf2B2S1', 'hex'],
         ['Ti2Ga1B2', 'hex'],
         ['Ti2Sn1B2', 'hex'],
         ['Al1B2Mo2', 'orth'],
         ['Hf2Tl1B2', 'hex'],
         ['Al1B2W2', 'orth'],
         ['Sc2Tl1B2', 'hex'],
         ['Zr2Al1B2', 'hex'],
         ['Zr2Cd1B2', 'orth'],
         ['Y2Cd1B2', 'orth'],
         ['Zr2Al1B2', 'orth'],
         ['Ti2Al1B2', 'orth'],
         ['Hf2B2Pb1', 'hex'],
         ['Nb2Ge1B2', 'hex']]
        """
        # all_data = pd.concat([train_data, val_data, test_data], axis=0, ignore_index=True)
        # print(all_data)

        # print(len(train_data))
        # print(len(test_data))
        # print(len(val_data))
        # target = all_data['target'].tolist()
        # predict = all_data['predict'].tolist()

        # target = test_data['target'].tolist()
        # predict = test_data['predict'].tolist()
        #
        # print(len(target))
        # print(len(predict))
