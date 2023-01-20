#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import os
from stability.StabilityAnalysis import EdAnalysis
from shutil import copyfile

here = os.path.abspath(os.path.dirname(__file__))

def main(threshold, label, stability_thresholds, space_group, main_label):
    path_to_ml_data = os.path.join(here, 'ml_data')

    process(path_to_ml_data, threshold, label, stability_thresholds, space_group, main_label)


def process(path_to_ml_data, threshold, label, stability_thresholds, space_group, main_label):
    """
    Args:
        training_prop (str) - 'Ef' if models trained on formation energies; 'Ed' if decomposition energies
        model (str) - ML model
        system (str) - 'allMP', 'LiMnTMO', or 'smact'
        path_to_ml_data (os.PathLike) - path to ml_data directory in .../TestStabilityML/mlstabilitytest/ml_data
    
    Returns:
        Runs all relevant analyses
        Prints a summary
    """
    if main_label:
        data_dir = os.path.join(path_to_ml_data, main_label + '_best_DNN')
    else:
        data_dir = os.path.join(path_to_ml_data, 'best_DNN')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_file = label + '_tgt_pred.json'  # {formula: Ed} to make: check_log.py run_best_target_pred()
    finput = os.path.join(data_dir, data_file)

    if not os.path.exists(finput):
        print('missing: %s' % data_file)
        return

    obj = EdAnalysis(data_dir,
                     data_file,
                     threshold)

    # obj.results(True)
    obj.results_summary(label=label, stability_thresholds=stability_thresholds, space_group=space_group)
    print('got results')
    return


if __name__ == '__main__':
    from data.train_label_info import for_params
    from find_best import best_models

    from data.data import id_hull_json

    # main_label = 'orth'
    main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    # space_group = 'hex_orth'
    # space_group = 'orth'
    space_group = 'hex'

    bests = best_models(main_label)
    # labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params([10, 20, 30, 40], [[600, 400], [400, 200], [100, 50],[200, 100, 40]])
    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(bests.fea_nums, bests.hls)
    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        new_label = opt_feas_num + '_' + label

        stability_thresholds = [0, 0.03, 0.07, 0.139, 0.15, 0.2, 0.26]

        for i in stability_thresholds:
            main(i, new_label, stability_thresholds, space_group, main_label)

    ### Then DFT_ML_predict_results.py and rename
    ### Then plot other pictures
