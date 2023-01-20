import pandas as pd
from utils.utils import read_json
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from plotter.plotter import yqsun_plot_font
from utils.utils import check_dir_existence
from find_best import best_models
from data.train_label_info import for_params


def get_price_df_for_n_m_feas(opt_num, tgt_pred, main_label):
    if main_label:
        opt_nmi_dir = main_label + '_opt_nmis'
        main_label_dir = main_label + '_figures'
        opt_feas_dir = main_label + '_opt_features'
    else:
        opt_nmi_dir = 'opt_nmis'
        main_label_dir = 'figures'
        opt_feas_dir = 'opt_features'

    print(opt_nmi_dir)
    print(main_label_dir)
    print(opt_feas_dir)

    pred_json = read_json(tgt_pred)  # run check_log.py
    print('opt_num: ', opt_num)
    opt_dir = os.path.join(r'features', opt_feas_dir)
    opt_feas_file_path = 'all_' + str(opt_num) + '_optimal_features'
    file_path = os.path.join(opt_dir, opt_feas_file_path + '.xlsx')

    if '.csv' in file_path:
        price_df = pd.read_csv(file_path, header=0)
    if '.xlsx' in file_path:
        price_df = pd.read_excel(file_path, header=0)

    price_df = price_df.set_index('id')
    print(pred_json.keys())

    for index in price_df.index.values:
        # print(type(index))
        price_df.loc[index, 'predict'] = pred_json[str(index)]['predict']

    return price_df


def plot_n_m_feas(m_feas, n_feas, name, opt_num, orentation, tgt_pred,
                  main_label, camps, alpha):
    spines_dict = ['left', 'right', 'top', 'bottom']
    EXT = '.jpg'
    label_size = 22
    # font params as dict
    params = yqsun_plot_font.__default_params__
    params.update({'font.size': '18', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)
    print(params)
    if main_label:
        main_label_dir = main_label + '_figures'
    else:
        main_label_dir = 'figures'

    FIG_DIR = os.path.join('plotter', main_label_dir, 'subplot_feas')
    check_dir_existence(FIG_DIR)

    feas_name_dict = read_json(os.path.join(main_label_dir, 'opt_feas_name.json'))
    # best_fea_simple = dict(zip(best_feas_simplename.values(), best_feas_simplename.keys()))
    best_fea_simple = {feas_name_dict[k]['simple']: k for k in feas_name_dict}
    best_feas_prettyname = {k: feas_name_dict[k]['pretty'] for k in feas_name_dict}
    m_feas = [best_fea_simple[m_fea] for m_fea in m_feas]
    n_feas = [best_fea_simple[n_fea] for n_fea in n_feas]
    price_df = get_price_df_for_n_m_feas(opt_num, tgt_pred, main_label)
    print(price_df)

    label_dH = r'$\Delta$' + r'$\it{H}$'
    label_units = r'$\/(\frac{eV}{atom})$'
    ylabel = label_dH + r'$_{d, prediction}$' + label_units

    # fig, axs = plt.subplots(n_row, n_col, figsize=(14, 8))
    figsize = (14, 8)
    m_num = len(m_feas)
    n_num = len(n_feas)
    sub_num = n_num

    if sub_num == 1:
        index = None
        figsize = (8, 6)
    if sub_num == 2:
        if orentation == 'h':
            index = [['a', 'b']]
        if orentation == 'v':
            index = [['a'], ['b']]
    if sub_num == 3:
        if orentation == 'h':
            index = [['a', 'b', 'c']]
        if orentation == 'v':
            index = [['a'], ['b'], ['c']]
    if sub_num == 4:
        if orentation == 'h':
            index = [['a', 'b'], ['c', 'd']]
        if orentation == 'v':
            index = [['a'], ['b'], ['c'], ['d']]
    if sub_num == 6:
        index = [['a', 'b'], ['c', 'd'], ['e', 'f']]

    if sub_num == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=800)
        ticks_size = 20
        cmaps = [camps] * sub_num
        cm_idx = 0
        for m_fea, n_fea in zip(m_feas, n_feas):
            # print(m_fea)
            # print(n_fea)
            x = price_df.loc[:, m_fea]
            y = price_df.loc[:, 'predict']
            c = price_df.loc[:, n_fea]
            rcParams.update(params)
            pcm = ax.scatter(x, y, marker='o', c=c,
                             cmap=cmaps[cm_idx], alpha=alpha,
                             )

            xlabel = best_feas_prettyname[m_fea]
            clabel = best_feas_prettyname[n_fea]

            ax.set_xlabel(xlabel, weight='bold', size=label_size)
            ax.set_ylabel(ylabel, weight='bold', size=label_size)
            # pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
            #                     cmap=cmaps[col])
            rcParams.update(params)
            if clabel == 'spacegroup number':
                cb = plt.colorbar(pcm, ax=ax, ticks=[65, 187],
                                  label=clabel,
                                  )
                cb.set_label(clabel, fontsize=20)
            else:
                cb = plt.colorbar(pcm, ax=ax,
                                  label=clabel,
                                  )
                cb.set_label(clabel, fontsize=20)
                # cb.tick_params(label_size=16)
            plt.tight_layout()
            plt.subplots_adjust()  # compress the right of the figure
            cm_idx += 1
        plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
        tk = plt.gca()
        for i in spines_dict:
            tk.spines[i].set_linewidth(1.5)
        fig.savefig(os.path.join(FIG_DIR, '{0}'.format(name) + EXT))
        plt.close()

    if sub_num != 1:
        fig, axs = plt.subplot_mosaic(index,
                                      constrained_layout=True, figsize=figsize, dpi=800)
        # print(axs.items())
        # cmaps = ['viridis', 'plasma', 'cividis', 'copper', 'summer', 'autumn', 'spring', 'winter', 'inferno',
        #          'twilight',
        #          'magma', 'cool', 'Blues', 'Greens', 'Purples', 'Reds', 'prism', ]
        cmaps = [camps] * sub_num
        cm_idx = 0
        for (label, ax), m_fea, n_fea in zip(axs.items(), m_feas, n_feas):
            # print(m_fea)
            # print(n_fea)
            x = price_df.loc[:, m_fea]
            y = price_df.loc[:, 'predict']
            c = price_df.loc[:, n_fea]
            rcParams.update(params)
            pcm = ax.scatter(x, y, marker='o', c=c,
                             cmap=cmaps[cm_idx], alpha=alpha,
                             )

            xlabel = best_feas_prettyname[m_fea]
            clabel = best_feas_prettyname[n_fea]

            ax.set_xlabel(xlabel, weight='bold', size=label_size)
            ax.set_ylabel(ylabel, weight='bold', size=label_size)
            # pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
            #                     cmap=cmaps[col])
            rcParams.update(params)
            if clabel == 'spacegroup number':
                plt.colorbar(pcm, ax=ax, label=clabel, ticks=[65, 187])
            else:
                plt.colorbar(pcm, ax=ax, label=clabel)
            trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
            ax.text(-0.2, 1.12, label, transform=ax.transAxes + trans,
                    fontsize='large', fontweight='bold', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
            cm_idx += 1
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None,
        #                     hspace=None)
        # plt.show()
        fig.savefig(os.path.join(FIG_DIR, '{0}'.format(name) + EXT))
        plt.close()


def main(main_label, camps, alpha):
    if main_label:
        DNN_dir = main_label + '_best_DNN'
    else:
        DNN_dir = 'best_DNN'

    best = best_models(main_label)
    hls = best.hls
    fea_nums = best.fea_nums
    # print(hls)
    # print(fea_nums)

    labels, activations, hidden_layers, optimizers, opt_feas_nums = for_params(fea_nums, hls)
    new_label = opt_feas_nums[0] + '_' + labels[0]
    tgt_pred_json = new_label + r'_tgt_pred.json'
    tgt_pred = os.path.join(r'ml_data', DNN_dir, tgt_pred_json)

    if main_label == 'hex_orth_space':
        # ### hex_orth_space test1
        # # m_feas = ['maxMN', 'maxMN', 'maxMN', 'maxMN' ]
        # m_feas = ['meanMN', 'maxMN', 'meanMN', 'maxMN', ]
        # n_feas = ['SN', 'maxNU', 'rNU', 'NU']
        # # n_feas = ['YD', 'maxNU', 'rNU', 'meanEn']
        # pic_name = 'test1'
        # plot_n_m_feas(m_feas=m_feas, n_feas=n_feas,
        #               name=pic_name, opt_num=opt_feas_nums[0], orentation='h',
        #               tgt_pred=tgt_pred, main_label=main_label)

        ### hex_orth_space test2
        # m_feas = ['maxMN', 'maxMN', 'maxMN', 'maxMN' ]
        # m_feas = ['meanMN', 'maxMN', 'meanMN', 'maxMN', ]
        # m_feas = ['meanMN']
        m_feas = ['maxMN']
        # n_feas = ['SN']
        n_feas = ['NU']
        # n_feas = ['Ff']
        # n_feas = ['Fs']
        # n_feas = ['YD']
        # n_feas = ['SN', 'SN','maxNU', 'NU']
        # n_feas = ['YD', 'maxNU', 'rNU', 'meanEn']
        # pic_name = 'test2'
        # pic_name = 'MN_SN'
        # pic_name = 'maxMN_SN'
        pic_name = 'maxMN_NU'
        # pic_name = 'maxMN_Ff'
        # pic_name = 'maxMN_Fs'
        # pic_name = 'maxMN_YD'
        # pic_name = 'MN_Ff'
        plot_n_m_feas(m_feas=m_feas, n_feas=n_feas,
                      name=pic_name, opt_num=opt_feas_nums[0], orentation='h',
                      tgt_pred=tgt_pred, main_label=main_label, camps=camps, alpha=alpha)

    # if main_label == 'hex':
    #     ### hex test1
    #     m_feas = ['maxMN',]
    #     # m_feas = ['maxMN', 'maxMN', 'maxMN', 'maxMN' ]
    #     # m_feas = ['minMN', 'minMN', 'maxMN', 'maxMN', ]
    #     # n_feas = ['maxMN', 'rNfV', 'meanEn', 'YD']
    #     # n_feas = [ 'minMN', 'meanEn', 'YD', 'Fd',]
    #     n_feas = ['meanEn']
    #     # n_feas = ['YD']
    #     # n_feas = ['BC']
    #     # pic_name = 'test1'
    #     # pic_name = 'test2'
    #     pic_name = 'maxMN_En'
    #     # pic_name = 'maxMN_YD'
    #     # pic_name = 'maxMN_BC'
    #     plot_n_m_feas(m_feas=m_feas, n_feas=n_feas,
    #                   name=pic_name, opt_num=opt_feas_nums[0], orentation='h',
    #                   tgt_pred=tgt_pred, main_label=main_label, camps=camps, alpha=alpha)
    #
    # if main_label == 'orth':
    #     ### orth test1
    #     m_feas = ['maxMN', ]
    #     # m_feas = ['minMN', 'minMN', 'maxMN', 'maxMN', ]
    #     # n_feas = ['maxMN', 'rNfV', 'meanEn', 'YD']
    #     # n_feas = ['BC',]
    #     n_feas = ['meanMN',]
    #     # n_feas = ['YD',]
    #     pic_name = 'max_meanMN'
    #     # pic_name = 'maxMN_YD'
    #     # pic_name = 'maxMN_BC'
    #     # pic_name = 'test1'
    #     # pic_name = 'test2'
    #     plot_n_m_feas(m_feas=m_feas, n_feas=n_feas,
    #                   name=pic_name, opt_num=opt_feas_nums[0], orentation='h',
    #                   tgt_pred=tgt_pred, main_label=main_label, camps=camps, alpha=alpha)


if __name__ == "__main__":
    cmapssss = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
                'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
                'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
                'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
                'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
                'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2',
                'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
                'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
                'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
                'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
                'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
                'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r',
                'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
                'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
                'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
                'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
                'viridis', 'viridis_r', 'winter', 'winter_r']

    # main_label = 'orth'
    # main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'
    camps = 'coolwarm'
    alpha = 0.9
    main_labels = ['orth', 'hex', 'hex_orth_space']
    # main_labels = ['hex_orth_space']
    for main_label in main_labels:
        main(main_label, camps, alpha)

    # m_feas = ['minMN', 'minMN', 'minMN', 'minMN', 'minMN', 'minMN', ]
    # n_feas = ['maxMN', 'rNfV', 'meanEn', 'YD', 'Fd', 'adAW']
    # pic_name = 'test_hex'
    # plot_n_m_feas(m_feas=m_feas, n_feas=n_feas,
    #               name=pic_name, opt_num=opt_feas_nums[0], orentation='h',
    #               tgt_pred=tgt_pred, main_label=main_label)

    # plot_n_m_feas(m_feas=['meanMN', 'meanMN', 'meanMN', 'meanMN'], n_feas=['meanNpU', 'AO', 'meanOPS1', 'minBL'],
    #               feas_name_dict=feas_name_dict, name='test5', opt_num=20, orentation='h',
    #               tgt_pred=r'ml_data\best_DNN\20_200_100_40_layer_1000_relu_tgt_pred.json')
    # plot_n_m_feas(m_feas=['meanMN', 'meanMN', 'meanMN', 'meanMN'], n_feas=['adAW', 'meanOPS2', 'NU', 'EE'],
    #               feas_name_dict=feas_name_dict, name='test6', opt_num=20, orentation='h',
    #               tgt_pred=r'ml_data\best_DNN\20_200_100_40_layer_1000_relu_tgt_pred.json')
    # plot_n_m_feas(m_feas=['meanMN', 'meanMN', 'meanMN', 'meanMN'], n_feas=['minBL', 'adAW', 'maxNf', 'EE'],
    #               feas_name_dict=feas_name_dict, name='test7', opt_num=20, orentation='h',
    #               tgt_pred=r'ml_data\best_DNN\20_200_100_40_layer_1000_relu_tgt_pred.json')
