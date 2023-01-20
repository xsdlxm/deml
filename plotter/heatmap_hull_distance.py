import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymatgen.io.vasp.inputs import Structure, Poscar
from utils.utils import read_json, write_json
from utils.Structure_lib import stlist, elements, get_best_formula, ele_num


def make_Hd_array(Hd_dict_list):
    resser = []
    for mec_prop in Hd_dict_list:
        _temp = []
        if mec_prop['set'] == 'test':
            hull = mec_prop['predict']
        else:
            hull = mec_prop['hull_distance']
        # print(hull)
        _temp.append(hull)
        resser.append(_temp)
    arr = np.array(resser)
    arr1 = arr.reshape(len(elements().A()), len(elements().M()))
    return arr1


def make_conner_label_mask_array(entry_dict_list):
    resser = []
    for mec_prop in entry_dict_list:
        _temp = []
        if mec_prop['set'] == 'test':
            mask = True
        else:
            mask = False
        # print(mask)
        _temp.append(mask)
        resser.append(_temp)
    arr = np.array(resser)
    arr1 = arr.reshape(len(elements().A()), len(elements().M()))
    return arr1


def to_DFT_ML_df(arr1):
    res_df = pd.DataFrame(arr1,
                          index=elements().A(),
                          columns=elements().M()
                          )
    return res_df


def sort_M_A_ele_list(ele_list):
    rule1 = {'Sc': 0, 'Y': 1, 'Ti': 2, 'Zr': 3, 'Hf': 4, 'V': 5, 'Nb': 6, 'Ta': 7, 'Cr': 8, 'Mo': 9, 'W': 10,
             'Mn': 11, 'Tc': 12, 'Fe': 13, 'Ru': 14, 'Co': 15, 'Rh': 16, 'Ni': 17}
    rule2 = {'Zn': 0, 'Cd': 2, 'Al': 3, 'Ga': 4, 'In': 5, 'Tl': 6, 'Si': 7, 'Ge': 8, 'Sn': 9, 'P': 10, 'As': 11,
             'S': 12, 'Cl': 13}
    new_list = []
    for i in rule1:
        r1 = []
        for k in ele_list:
            M = k['M']
            if M == i:
                r1.append(k)
        r2 = []
        for j in rule2:
            for k in r1:
                if k["A"] == j:
                    r2.append(k)
        for k in r2:
            new_list.append(k)
    return new_list


def sort_A_M_ele_list(ele_list):
    rule1 = {'Sc': 0, 'Y': 1, 'Ti': 2, 'Zr': 3, 'Hf': 4, 'V': 5, 'Nb': 6, 'Ta': 7, 'Cr': 8, 'Mo': 9, 'W': 10,
             'Mn': 11, 'Tc': 12, 'Fe': 13, 'Ru': 14, 'Co': 15, 'Rh': 16, 'Ni': 17}

    rule2 = {'Zn': 0, 'Cd': 2, 'Al': 3, 'Ga': 4, 'In': 5, 'Tl': 6, 'Si': 7, 'Ge': 8, 'Sn': 9, 'Pb': 10, 'P': 11,
             'As': 12,
             'S': 13}

    new_list = []
    for i in rule2:
        r2 = []
        for k in ele_list:
            A = k['A']
            if A == i:
                r2.append(k)
        r1 = []
        for j in rule1:
            for k in r2:
                if k["M"] == j:
                    r1.append(k)
        for k in r1:
            new_list.append(k)
    return new_list


def _conner_label_mask(data, mask):
    if mask is None:
        mask = np.zeros(data.shape, bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
                and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask


def heatmapplotter(df, cmap, title='Formation energy of 212-MABs', label='formation energy (eV/atom)',
                   png_name='Ef_heatmap',
                   png_dir=r'G:\high-throughput-workflow\allthrough_heatmap\fomation_energy_heatmap',
                   mask_df=None,
                   show_title=False):
    from plotter.plotter import yqsun_plot_font
    from matplotlib import rcParams
    ### heatmap plotter ---------------------------------------------------
    params = yqsun_plot_font.__default_params__
    params.update({'font.size': '16', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)
    title_size = 22
    ### heatmap plotter ---------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=800)
    if show_title == True:
        plt.title(title, size=title_size, weight='bold')
    sns.heatmap(data=df,
                # vmin=5,  # 图例（右侧颜色条color bar）中最小显示值
                # vmax=8,  # 图例（右侧颜色条color bar）中最大显示值
                cmap=plt.get_cmap(cmap),  # 使用matplotlib中的颜色盘 'Set3', 'tab20c', 'Greens', 'Greens_r',
                # center=0.4,  # color bar的中心数据值大小，可以控制整个热图的颜盘深浅
                annot=True,  # 默认为False，当为True时，在每个格子写入data中数据
                fmt=".3f",  # 设置每个格子中数据的格式，参考之前的文章，此处保留两位小数
                annot_kws={'size': 7, 'weight': 'bold'},  # 设置格子中数据的大小、粗细、颜色
                # linewidths=1,  # 每个格子边框宽度，默认为0
                # linecolor='red',  # 每个格子边框颜色,默认为白色
                cbar=True,  # 右侧图例(color bar)开关，默认为True显示
                cbar_kws={'label': label,  # color bar的名称
                          # 'orientation': 'vertical',  # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                          'orientation': 'horizontal',  # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                          # "ticks": np.arange(4.5, 8, 0.5),  # color bar中刻度值范围和间隔
                          "format": "%.1f",  # 格式化输出color bar中刻度值
                          "pad": 0.08,  # color bar与热图之间距离，距离变大热图会被压缩
                          },
                # mask=df < 0,  # 热图中显示部分数据：显示数值小于6的数据
                # xticklabels=['三连啊', '关注公众号啊', 'pythonic生物人', '收藏啊', '点赞啊', '老铁三连三连'],
                # # x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
                # yticklabels=True,  # y轴方向刻度标签开关、同x轴
                conner_label=True,
                conner_label_mask=mask_df
                )

    ###
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(r'G:\high-throughput-workflow\allthrough_heatmap\hull_heatmap_%s' % str(cmaps.index(cmap)))
    plt.savefig(png_dir + r'/' + png_name + '_%s' % cmap)
    plt.close()


def analyzer(threshold, meta_threshold):
    import pprint
    ### read convex hull distance json to dict
    hull_dict = hull_out(threshold=meta_threshold)
    # dft_Ef = read_json(r'G:\high-throughput-workflow\Static_wf\calculation_formation_energy\dft_Ef_json')
    # exf = read_json(r'G:\high-throughput-workflow\MBenes_Eexf\Eexf\all_exf_json').get_json_to_dict()
    # print(hull_dict)

    stable_dict = {}
    meta_stable_dict = {}
    unstable_dict = {}

    for k in hull_dict:
        if hull_dict[k]['stability'] == True and hull_dict[k]['hull_distance'] < threshold:
            stable_dict[k] = hull_dict[k]

        if hull_dict[k]['stability'] == True and threshold < hull_dict[k]['hull_distance'] < meta_threshold:
            meta_stable_dict[k] = hull_dict[k]
        else:
            unstable_dict[k] = hull_dict[k]

    print('stable with threshold %s: ' % threshold)
    print(len(stable_dict))
    pprint.pprint([k for k in stable_dict])
    print('meta-stable from %s to %s: ' % (threshold, meta_threshold))
    print(len(meta_stable_dict))
    pprint.pprint([k for k in meta_stable_dict])
    print('unstable with meta-threshold %s: ' % meta_threshold)
    print(len(unstable_dict))


def main(main_label, space_group, show_title=False):
    png_name = main_label + '_Hd_heatmap_' + space_group
    dir = 'G:\codes\modnet\mab_ml\data\mp_opted_structures_' + space_group
    ### read structures path list -------------------------------------------
    if space_group == 'orth':
        title = 'Decomposition energies of orthorhombic M' + r'$_{2}$' + 'AB' + r'$_{2}$'
    if space_group == 'hex':
        title = 'Decomposition energies of hexagonal M' + r'$_{2}$' + 'AB' + r'$_{2}$'

    st_list = [os.path.join(dir, f) for f in os.listdir(dir)]
    ### elements needed
    M_ele = elements().M()
    A_ele = elements().A()

    # from data.data import hull_out, id_hull_json
    # analyzer(threshold=0.03, meta_threshold=0.15, space_group=space_group)
    # list_of_hull = id_hull_json(space_group).values()
    id_json_path = get_DFT_ML_predict_results_id_json(main_label=main_label)
    # print(id_json_path)
    model_id_json = read_json(id_json_path)
    MAB_entry_dict_list = [model_id_json[k] for k in model_id_json if model_id_json[k]["space_group"] == space_group]
    # MAB_entry_dict_list = [d for d in MAB_entry_dict_list if d["set"] == 'test']
    # print(space_group_model_id_json)
    # MAB_entry_dict_list = model_id_json.values()
    print(len(MAB_entry_dict_list))
    # print(len(MAB_entry_dict_list))

    sorted_dict_list = sort_A_M_ele_list(list(MAB_entry_dict_list))
    # print(sorted_dict_list)
    arr1 = make_Hd_array(sorted_dict_list)
    arr2 = make_conner_label_mask_array(sorted_dict_list)
    ### sorted_dict_list to df for deltaHd heatmap
    df1 = to_DFT_ML_df(arr1)
    df2 = to_DFT_ML_df(arr2)
    # df1 = to_DFT_ML_df(sorted_dict_list, _type='hull_distance')
    # print(df1)
    # print(df2)

    ### plot deltaHd heatmap with df
    # aa = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
    # aa = ['coolwarm', 'cool', 'Blues_r']
    aa = ['coolwarm']
    for cmap in aa:
        heatmapplotter(df1, cmap, title=title,
                       label=r'$\Delta$' + r'$\it{H}$' + r'$_{d}$' + r'$\/(\frac{eV}{atom})$',
                       png_name=png_name,
                       png_dir=r'figures',
                       mask_df=df2,
                       show_title=show_title)

    #     # heatmapplotter(df, cmap, title='Formation energies of 212-MABs', label='formation energy (eV/atom)',
    #     #                png_name='Ef_heatmap',
    #     #                png_dir=r'G:\high-throughput-workflow\allthrough_heatmap\fomation_energy_heatmap')
    #
    #     # heatmapplotter(df2, cmap, title='Exfoliation energy of 212-MABs', label='exfoliation energy (eV/atom)',
    #     #                png_name='Eexf_heatmap',
    #     #                png_dir=r'G:\high-throughput-workflow\allthrough_heatmap\exfoliation_energy')


if __name__ == '__main__':
    from utils.utils import check_dir_existence
    from data.data import get_DFT_ML_predict_results_id_json

    check_dir_existence('figures')
    # main_labels = ['hex', 'orth', 'hex_orth_space']
    # space_groups = ['orth', 'hex']
    main('hex_orth_space', 'hex', show_title=False)
    main('hex_orth_space', 'orth', show_title=False)
    main('orth', 'orth', show_title=False)
    main('hex', 'hex', show_title=False)
