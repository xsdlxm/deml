#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xsdlxm
"""
import os
import pandas as pd
from utils.utils import read_json
from shutil import unpack_archive

this_dir, this_filename = os.path.split(__file__)


def spaces():
    """
    {compound (str) : chemical space containing compound that is easy to eval (str)
        for compound in MP}
    """
    fjson = os.path.join(this_dir, "data", "spaces.json")
    return read_json(fjson)


def hull_out(threshold=0.1, space_group=None):
    """
    :param threshold:
    :return: {"34072651": {"pretty_formula": "Al(BMo)2", "formula": "Al1B2Mo2",
    "comps": ["AlBMo", "BMo"], "id": "34072651", "hull_distance": 0.04932791399999914,
    "formation_energy": -0.42149445333333374, "space_group_number": 187,
    "space_group": "hex", "A": "Al", "M": "Mo", "stability": True}, ........},
    """
    fjson = id_hull_json(space_group=space_group)
    for k in fjson:
        _dict = fjson[k]
        Ed = _dict["hull_distance"]
        Ef = _dict["formation_energy"]
        if Ed <= threshold:
            fjson[k]["stability"] = True
        else:
            fjson[k]["stability"] = False
        fjson[k]['Ef'] = Ef
        fjson[k]['Ed'] = Ed
        fjson[k]['rxn'] = _dict["comps"]

    return fjson

def id_hull_json(space_group='hex_orth'):
    """
    :param space_group: 'both'/'orth'/'hex'
    :return: {"34072651": {"pretty_formula": "Al(BMo)2", "formula": "Al1B2Mo2",
    "comps": ["AlBMo", "BMo"], "id": "34072651", "hull_distance": 0.04932791399999914,
    "formation_energy": -0.42149445333333374, "space_group_number": 187,
    "space_group": "hex", "A": "Al", "M": "Mo"},
    """
    fjson = {}
    if space_group == 'hex_orth':
        fjson_hex = read_json(os.path.join(this_dir, "data", "id_CV_outputs_hex.json"))
        fjson_orth = read_json(os.path.join(this_dir, "data", "id_CV_outputs_orth.json"))
        fjson.update(fjson_hex)
        fjson.update(fjson_orth)
        print('_______________________________________________')
    else:
        fjson = read_json(os.path.join(this_dir, "data", "id_CV_outputs_" + space_group + ".json"))

    return fjson

def id_model_hull_json(main_label):
    if main_label == 'hex':
        space_group = 'hex'
    if main_label == 'orth':
        space_group = 'orth'
    if main_label == 'hex_orth_space':
        space_group = 'hex_orth'
    if main_label == 'hex_orth':
        space_group = 'hex_orth'

    fjson = read_json(os.path.join(this_dir, "data", json_name))
    return fjson

def Ed_csv():
    df = pd.read_csv(this_dir, "data", "mab_Ed.csv")
    return df

def get_DFT_ML_predict_results_id_json(main_label):
    from mab_ml.find_best import best_models
    from mab_ml.data.train_label_info import for_params
    best = best_models(main_label)
    hls = best.hls
    fea_nums = best.fea_nums
    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(fea_nums, hls)
    print(labels)
    label = labels[0]
    opt_feas_num = opt_feas_nums[0]
    module_label = opt_feas_num + '_' + label
    id_json_path = os.path.join(this_dir, "data", "%s_%s_set_id.json" % (main_label, module_label))
    return id_json_path

def ml_out(label):

    if label:
        label_name = label+"_ml_output.json"
    else:
        label_name = "ml_output.json"
    fjson = os.path.join(this_dir, "data", label_name)
    return read_json(fjson)


if __name__ == '__main__':
    import pprint

    # id_hull = id_hull_json()
    # print(id_hull)
    # h_out = hull_out()
    # pprint.pprint(h_out)
