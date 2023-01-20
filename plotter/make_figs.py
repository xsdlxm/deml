#   coding:utf-8
__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

from plotter.plotter import plt_mse, plot_tgt_pred_mae, confusion_matrix_plotter, n_feas_error_plotter, heatmapplotter
from data.train_label_info import for_params
from data.data_analyzer import read_id_tgt_pred_data, read_test_loss, read_mse_2data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_data_for_plt(fn, has_first_column=True):
    if has_first_column:
        star_num = 0
    else:
        star_num = 1
    ori_data = pd.read_csv(fn)
    column = ori_data.columns.values.tolist()[star_num:-2]
    data = ori_data.values
    train_data = data[:, star_num:-2]
    dd = np.corrcoef(train_data, rowvar=0)
    # print(dd.shape)
    return dd, column


def plt_cross_nmi():
    label_font = {"fontsize": 14, 'family': 'Times New Roman'}
    index_label_font = {"fontsize": 18, 'weight': 'light', 'family': 'Times New Roman'}
    tick_font_size = 14
    tick_font_dict = {"fontsize": 14, 'family': 'Times New Roman'}
    fig = plt.figure(figsize=(9, 4))
    plt.rc('font', family='Times New Roman', weight='normal')
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    ax = plt.subplot2grid((1, 23), (0, 0), colspan=10, rowspan=1, fig=fig)
    ax2 = plt.subplot2grid((1, 23), (0, 10), colspan=1, rowspan=1, fig=fig)
    ax2.tick_params(axis='both', labelsize=tick_font_size - 2)
    ax3 = plt.subplot2grid((1, 23), (0, 13), colspan=10, rowspan=1, fig=fig)
    head_dir = r"G:\ztml\ztml\rdata\all_rmcoref_data"

    csv_file = os.path.join(head_dir, r'temp_clean_data.csv')
    dd, columns = read_data_for_plt(csv_file, has_first_column=False)

    column = []
    for nn in columns:
        if str(nn).startswith('K') or str(nn).startswith('k'):
            _label = r'$\kappa_{%s}$' % nn[1:]
        elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith('m'):
            _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
        elif str(nn) == 'a_b':
            _label = 'a/b'
        elif str(nn) == 'NC.1':
            nn = "NCo"
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        else:
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        column.append(_label)

    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar_ax=ax2,
                    cbar_kws={"ticks": np.arange(1, -1.2, -0.2)})
    ax.set_xticks(np.array(range(0, len(column))))
    ax.set_xlim(0, len(column))
    ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)

    ax.set_yticks(np.array(range(0, len(column))))
    ax.set_ylim(0, len(column))
    ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([column[i] if i % 2 == 0 else None for i in range(len(column))],
                       fontdict=tick_font_dict, minor=True,
                       rotation=85)
    ax.set_yticklabels([column[i] if i % 2 == 0 else None for i in range(len(column))], fontdict=tick_font_dict,
                       minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))

    ax.grid(alpha=0.7, linewidth=0.1, color='gray')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax.tick_params(axis='x', direction='in', labelrotation=85, length=0.00001)
    ax.tick_params(axis='y', direction='in', labelrotation=0, length=0.00001)
    # ax.tick_params(axis='y', labelrotation=-45)
    ax.text(0.1, 31.7, '(a)', fontdict=index_label_font)

    csv_file = os.path.join(head_dir, r'normalized_data.csv')
    dd, columns = read_data_for_plt(csv_file, has_first_column=True)
    column = []
    for nn in columns:
        if str(nn).startswith('K') or str(nn).startswith('k'):
            _label = r'$\kappa_{%s}$' % nn[1:]
        elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith('m'):
            _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
        elif str(nn) == 'a_b':
            _label = 'a/b'
        elif str(nn) == 'NC.1':
            nn = "NCo"
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        elif len(str(nn)) == 1:
            _label = str(nn)
        else:
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        column.append(_label)

    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax3, cbar=False)
    ax3.set_xticks(np.array(range(len(column))))
    ax3.set_xlim(0, len(column))
    ax3.set_xticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_yticks(np.array(range(len(column))))
    ax3.set_ylim(0, len(column))
    ax3.set_yticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([i for i in column], fontdict=tick_font_dict, minor=True, rotation=85)
    ax3.set_yticklabels([i for i in column], fontdict=tick_font_dict, minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))

    ax3.grid(alpha=0.7, linewidth=0.1, color='gray')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax3.tick_params(axis='x', direction='in', labelrotation=85, length=0.00001)
    ax3.tick_params(axis='y', direction='in', labelrotation=0, length=0.00001)
    ax3.text(0.1, 11.3, '(b)', fontdict=index_label_font)

    # plt.savefig('plt_coref_fig1.pdf')
    # plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.12, top=0.93, right=0.98, wspace=1.2)
    # plt.show()
    plt.savefig('plt_coref_fig1.pdf', dpi=600)
    plt.savefig('plt_coref_fig1.jpg', dpi=600)
    plt.savefig('plt_coref_fig1.tiff', dpi=600)


if __name__ == '__main__':
    import os
    from utils.utils import check_dir_existence

    main_label = 'orth'
    # main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    if main_label:
        main_label_dir = main_label + '_save_dir'
        if main_label == 'hex':
            space_group = 'hex'
        if main_label == 'orth':
            space_group = 'orth'
        if main_label == 'hex_orth_space':
            space_group = 'hex_orth'
        if main_label == 'hex_orth':
            space_group = 'hex_orth'
    else:
        main_label_dir = 'save_dir'
    # print(space_group)

    train_save_dir = os.path.join(r'..\train', main_label_dir)
    test_save_dir = os.path.join(r'..\test', main_label_dir)

    check_dir_existence(train_save_dir)
    check_dir_existence(test_save_dir)

    from find_best import best_models

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
        test_data = read_id_tgt_pred_data(test_out)
        train_val_mseloss = read_mse_2data(train_log)
        train_data = read_id_tgt_pred_data(train_out)
        val_data = read_id_tgt_pred_data(val_out)
        all_data = pd.concat([train_data, val_data, test_data], axis=0, ignore_index=True)
        # print(all_data)

        # print(len(train_data))
        # print(len(test_data))
        # print(len(val_data))
        # target = all_data['target'].tolist()
        # predict = all_data['predict'].tolist()
        # set = 'train'
        # set = 'test'
        set = 'validation'

        if set:
            if set == 'test':
                target = test_data['target'].tolist()
                predict = test_data['predict'].tolist()
            if set == 'validation':
                target = val_data['target'].tolist()
                predict = val_data['predict'].tolist()
            if set == 'train':
                target = train_data['target'].tolist()
                predict = train_data['predict'].tolist()
            if set == 'all':
                target = all_data['target'].tolist()
                predict = all_data['predict'].tolist()

        # print(len(target))
        # print(len(predict))
        # print(set)
        # print(target)
        # print(predict)

        plot_tgt_pred_mae(target, predict, 'Ed', opt_feas_num, hidden_layer, module_label, main_label, set)
        train_val_mseloss = train_val_mseloss.values
        plt_mse(train_val_mseloss, opt_feas_num, hidden_layer, module_label, main_label, show=False)

        if set:
            thresholds = [0.07]
        else:
            thresholds = [0, 0.03, 0.1, 0.15, 0.2]
        # set = 'test' or 'validation'
        set = 'test'
        for i in thresholds:
            # print(space_group)
            confusion_matrix_plotter('Ed', i, opt_feas_num, hidden_layer, module_label, space_group, main_label, set)
