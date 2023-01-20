import os
import matplotlib.pyplot as plt
from pylab import mpl
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from data.data import hull_out
from utils.utils import read_json, write_json
from math import ceil, floor

spines_dict = ['left', 'right', 'top', 'bottom']


def tableau_colors():
    """
    Args:

    Returns:
        dictionary of {color (str) : RGB (tuple) for the dark tableau20 colors}
    """
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    names = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'turquoise']
    colors = [tableau20[i] for i in range(0, 20, 2)]
    return dict(zip(names, colors))


def set_rc_params():
    """
    Args:

    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {'axes.linewidth': 1.5,
              'axes.unicode_minus': False,
              'figure.dpi': 300,
              'font.size': 20,
              'legend.frameon': False,
              'legend.handletextpad': 0.4,
              'legend.handlelength': 1,
              'legend.fontsize': 12,
              'mathtext.default': 'regular',
              'savefig.bbox': 'tight',
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'xtick.major.size': 6,
              'ytick.major.size': 6,
              'xtick.major.width': 1.5,
              'ytick.major.width': 1.5,
              'xtick.top': True,
              'ytick.right': True,
              'axes.edgecolor': 'black',
              'figure.figsize': [6, 4]}
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


def get_results(label, model_label):
    if label:
        DNN_dir = label + '_best_DNN'
    else:
        DNN_dir = 'best_DNN'

    ml_results = model_label + '_ml_results.json'

    return read_json(os.path.join(r'ml_data', DNN_dir, ml_results))


def get_compounds(space_group):
    d = hull_out(space_group=space_group)
    return sorted(list(d.keys()))


def get_actual(prop, compounds, space_group):
    d = hull_out(space_group=space_group)
    return [d[c][prop] for c in compounds]


def get_pred(prop, label, model_label):
    results = get_results(label, model_label)
    d = dict(zip(results['data']['ids'], results['data'][prop]))
    return [d[c] for c in d]


def plt_mse(training_data, fea_num=20, hl='', model_label=None, main_label=None, show=False):
    tdata = training_data
    if main_label:
        main_label_dir = main_label + '_figures'
    else:
        main_label_dir = 'figures'

    FIG_DIR = os.path.join('plotter', main_label_dir)
    EXT = '.jpg'
    title_font = {"fontsize": 24, }
    pad = 20
    label_font = {"fontsize": 22, }
    text_font = {"fontsize": 18, }
    xpos, ypos = 365, 0.015
    ticks_size = 20
    linewidth = 3.0
    legend_size = 18
    figsize = (8, 6)
    dpi = 800
    # font params as dict
    params = yqsun_plot_font.__default_params__
    params.update({'font.weight': 'bold'})
    # print(params)
    # # set params to rcParams (control the font)
    rcParams.update(params)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # plt.rcParams["xtick.direction"] = 'in'
    # plt.rcParams["ytick.direction"] = 'in'

    # left, bottom, width, height = 1/ncol*0.66 * (i+1), 1/nrow * 1.26 * (int(i / ncol) + 1), 0.125, 0.12
    # left, bottom, width, height = d.x0 + d.width * 1/ncol, d.y0+0.12/nrow, 0.125, 0.12
    tx = tdata[:, 0]
    ytrain = tdata[:, -2]
    ytest = tdata[:, -1]

    # ax2 = fig.add_axes([left, bottom, width, height])
    ylabel = 'MSE' + r'$\/(\frac{eV}{atom})$'
    xlabel = 'Epoch'
    title = 'Training and validation MSEloss vs Epoch'

    xlim = (-10, 1000)
    ylim = (-0.001, 0.1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, (0.0, 0.02, 0.04, 0.06, 0.08, 0.10))
    plt.title(title, fontdict=title_font, weight='bold', pad=pad)
    plt.plot(tx, ytrain, c='#347FE2', linewidth=linewidth, label='Training Set')
    plt.plot(tx, ytest, c='#F37878', linewidth=linewidth, label='Validation Set')
    num1, num2, num3, num4 = 0.50, 0.76, 3, 0
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, title=None, fontsize=legend_size)
    # train_final_mean = np.mean(ytrain)
    train_final_mean = ytrain[-1]
    # test_final_mean = np.mean(ytest)
    test_final_mean = ytest[-1]
    text = "Feature number : %s\n" % str(
        fea_num) + "Hidden Layers : %s\n" % hl + "Activation : Relu\nOptimizer : Adam\n" + "Training MSE:%.4f\nValidation MSE:%.4f" % (
               train_final_mean, test_final_mean)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.text(xpos, ypos, text, fontdict=text_font)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.89,
                        bottom=0.16,
                        left=0.15,
                        right=0.85,
                        hspace=0.2,
                        wspace=0.2)

    if show:
        plt.show()

    savename = 'Fig_mse_' + model_label + '%s' % EXT
    fig.savefig(os.path.join(FIG_DIR, savename))
    # fig.savefig(os.path.join(FIG_DIR, 'test' + savename))
    plt.close()


def ax_generic_scatter(x, y,
                       alpha=0.1,
                       marker='o',
                       lw=0,
                       s=10,
                       colors='black',
                       edgecolors=None,
                       vmin=False,
                       vmax=False,
                       cmap=False,
                       cmap_values=False,
                       xticks=(True, (0, 1)),
                       yticks=(True, (0, 1)),

                       xlabel='x',
                       ylabel='y',
                       label_size=11,
                       ticks_size=11,
                       xlim=(0, 1),
                       ylim=(0, 1),
                       diag=('black', 1, '--')):
    if colors == 'cmap':
        cm = plt.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.scatter(x, y,
                    c=cmap_values,
                    cmap=cm,
                    norm=norm,
                    edgecolors=edgecolors,
                    alpha=alpha,
                    marker=marker,
                    lw=lw,
                    s=s,
                    rasterized=True)
    else:
        if isinstance(colors, list):
            plt.scatter(x, y,
                        c=colors,
                        edgecolors=edgecolors,
                        alpha=alpha,
                        marker=marker,
                        lw=lw,
                        s=s)
        else:
            plt.scatter(x, y,
                        color=colors,
                        edgecolor=edgecolors,
                        alpha=alpha,
                        marker=marker,
                        lw=lw,
                        s=s)
    # print(ticks_size)
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params()
    if not xticks[0]:
        plt.gca().xaxis.set_ticklabels([])
    if not yticks[0]:
        plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(xlabel, fontdict={'size': label_size}, weight='bold')
    plt.ylabel(ylabel, fontdict={'size': label_size}, weight='bold')
    if diag:
        plt.plot(xlim, xlim, color=diag[0], lw=diag[1], ls=diag[2])
    plt.xlim(xlim)
    plt.ylim(ylim)


def ax_actual_vs_pred(actual, pred, prop, text, text_font, label_size, ticks_size, s=25,
                      show_xticks=True, show_yticks=True,
                      show_xlabel=True, show_ylabel=True,
                      show_mae=True, show_model=True):
    x, y = pred, actual

    alpha = 0.9
    marker = 'o'
    lw = 1

    colors = 'cmap'
    edgecolors = None
    # cmap = 'plasma_r'
    cmap = 'coolwarm'
    cmap_values = [abs(actual[i] - pred[i]) for i in range(len(actual))]
    vmin, vmax = 0, max(cmap_values) + 0.01
    diag = ('blue', 1, '--')

    # min_ = min(min(y), min(x)) -0.1
    # max_ = max(max(y), max(x)) + 0.1
    # ticks = (min_, 0, max_)
    ticks = (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1)
    xtick_values, ytick_values = ticks, ticks

    if show_xlabel or show_ylabel:
        label_dH = r'$\Delta$' + r'$\it{H}$'
        label_units = r'$\/(\frac{eV}{atom})$'
        if prop == 'Ef':
            xlabel = label_dH + r'$_{f, prediction}$' + label_units
            ylabel = xlabel.replace('prediction', 'DFT')
        elif prop == 'Ed':
            xlabel = label_dH + r'$_{d, prediction}$' + label_units
            ylabel = xlabel.replace('prediction', 'DFT')
    if not show_xlabel:
        xlabel = ''
    if not show_ylabel:
        ylabel = ''

    xlim = (-0.2, 1)
    ylim = xlim

    ax = ax_generic_scatter(x, y,
                            alpha=alpha,
                            marker=marker,
                            lw=lw,
                            s=s,
                            colors=colors,
                            edgecolors=edgecolors,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_values=cmap_values,
                            xticks=(show_xticks, xtick_values),
                            yticks=(show_yticks, ytick_values),
                            xlabel=xlabel,
                            ylabel=ylabel,
                            label_size=label_size,
                            ticks_size=ticks_size,
                            xlim=xlim,
                            ylim=ylim,
                            diag=diag)

    if show_model:
        x_range = xlim[1] - xlim[0]
        x_offset, y_offset = 0.08, 0.12
        ax = plt.text(xlim[0] + x_offset * x_range, xlim[1] - y_offset * x_range, show_model, fontdict={'size': 10})
    if show_mae:
        xpos, ypos = -0.17, 0.32

        from data.data_analyzer import data_linregress
        r_2, rmse, accuracy, mae, mae_accuracy = data_linregress(actual, pred)
        # print(rmse)
        # print(mae)

        text = text % (r_2, rmse, mae, 100 * mae_accuracy) + "%\n"

        plt.text(xpos, ypos, text,
                 fontdict={'size': text_font, 'weight': 'bold'},
                 horizontalalignment='left',
                 verticalalignment='bottom'
                 )

    plt.xlim(xlim)
    plt.ylim(ylim)
    return ax


def add_colorbar(fig, label, ticks,
                 cmap, vmin, vmax, position,
                 label_size, tick_size, tick_len, tick_width):
    import matplotlib as mpl

    cax = fig.add_axes(position)

    cmap = getattr(plt.cm, cmap)
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax,
                      ticks=ticks,
                      orientation='vertical')
    cb.set_label(label, fontsize=label_size, weight='bold')
    cb.set_ticks(ticks)
    # cb.ax.set_yticklabels([])
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    return fig


def plot_tgt_pred_mae(actual, pred, prop, fea_num, hl, module_label, label, set):
    if label:
        label_dir = label + '_figures'
    else:
        label_dir = 'figures'
    title_size = 24
    pad = 20
    label_size = 22
    ticks_size = 20
    s = 25
    cticks = (0, 0.1, 0.2, 0.3)
    bar_label = '| DFT - prediction |' + r'$\/(\frac{eV}{atom})$'
    bar_label_size = 18
    bar_tick_size = 18
    text_font = 18
    figsize = (8, 6)
    dpi = 800
    FIG_DIR = os.path.join('plotter', label_dir)
    EXT = '.jpg'
    fea_num = str(fea_num)
    hl = str(hl)
    text = "Feature number : %s\n" % fea_num + "Hidden Layers : %s\n" % hl + "R-squared(R2) : %.4f\nActivation : Relu\nOptimizer : Adam\nRMSE : %.4f\nMAE : %.4f\nAccuracy : %.2f"
    # font params as dict
    params = yqsun_plot_font.__default_params__
    params.update({'font.weight': 'bold'})
    # print(params)
    # # set params to rcParams (control the font)
    rcParams.update(params)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title('Regression perfomace', fontdict={'size': title_size, 'weight': 'bold'}, pad=pad)
    ax = ax_actual_vs_pred(actual, pred, prop, text, text_font=text_font, label_size=label_size, ticks_size=ticks_size,
                           s=s,
                           show_xticks=True, show_yticks=True,
                           show_xlabel=True, show_ylabel=True,
                           show_mae=True, show_model=False)
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)

    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)

    # cmap_values = [abs(actual[i] - pred[i]) for i in range(len(actual))]
    # vmin, vmax = 0, max(cmap_values) + 0.01
    vmin, vmax = 0, 0.3
    add_colorbar(fig,
                 bar_label,
                 cticks,
                 # 'plasma_r',
                 'coolwarm',
                 vmin, vmax,
                 [0.86, 0.24, 0.015, 0.6],
                 bar_label_size, bar_tick_size, 1, 2)
    # plt.tight_layout()

    plt.subplots_adjust(top=0.90,
                        bottom=0.17,
                        left=0.15,
                        right=0.85,
                        hspace=0.2,
                        wspace=0.2)

    # plt.show()
    savename = set + '_Fig_tgt_pred_mae_' + module_label + '%s' % EXT
    fig.savefig(os.path.join(FIG_DIR, savename))
    # fig.savefig(os.path.join(FIG_DIR, 'test' + savename))
    plt.close()


def ax_x_vs_y(fea_x, pred, prop,
              show_xticks=True, show_yticks=True,
              show_xlabel=True, show_ylabel=True,
              show_mae=True, show_model=True):
    x, y = pred, fea_x

    alpha = 0.9
    marker = 'o'
    lw = 4
    s = 25  # 点大小
    tick_size = 6
    colors = 'cmap'
    edgecolors = None
    # cmap = 'plasma_r'
    cmap = 'coolwarm'
    cmap_values = [abs(fea_x[i] - pred[i]) for i in range(len(fea_x))]
    vmin, vmax = 0, max(cmap_values) + 0.01
    diag = ('blue', 1.4, '--')

    # min_ = min(min(y), min(x)) -0.1
    # max_ = max(max(y), max(x)) + 0.1
    # ticks = (min_, 0, max_)
    ticks = (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1)
    xtick_values, ytick_values = ticks, ticks

    if show_xlabel or show_ylabel:
        label_dH = r'$\Delta$' + r'$\it{H}$'
        label_units = r'$\/(\frac{eV}{atom})$'
        if prop == 'Ef':
            xlabel = label_dH + r'$_{f,pred}$' + label_units
            ylabel = xlabel.replace('pred', 'DFT')
        elif prop == 'Ed':
            xlabel = label_dH + r'$_{d,pred}$' + label_units
            ylabel = xlabel.replace('pred', 'DFT')
    if not show_xlabel:
        xlabel = ''
    if not show_ylabel:
        ylabel = ''

    xlim = (-0.2, 1)
    ylim = xlim

    ax = ax_generic_scatter(x, y,
                            alpha=alpha,
                            marker=marker,
                            lw=lw,
                            s=s,
                            colors=colors,
                            edgecolors=edgecolors,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_values=cmap_values,
                            xticks=(show_xticks, xtick_values),
                            yticks=(show_yticks, ytick_values),
                            tick_size=tick_size,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            xlim=xlim,
                            ylim=ylim,
                            diag=diag)

    if show_model:
        x_range = xlim[1] - xlim[0]
        x_offset, y_offset = 0.08, 0.12
        ax = plt.text(xlim[0] + x_offset * x_range, xlim[1] - y_offset * x_range, show_model, fontdict={'size': 20})
    if show_mae:
        xpos, ypos = 0.5, -0.2

        from data.data_analyzer import data_linregress
        r_2, rmse, accuracy, mae, mae_accuracy = data_linregress(fea_x, pred)

        text_font = 20

    plt.xlim(xlim)
    plt.ylim(ylim)
    return ax


def plot_feature_x_y_tgt(fea_x, target, prop, model, hl):
    params = yqsun_plot_font.__default_params__
    # print(params)
    params.update({'font.size': '13', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    fig = plt.figure(figsize=(8, 6), dpi=400)
    # FIG_DIR = os.path.join('plotter', 'figures')
    # EXT = '.jpg'
    # hl = str(hl)
    # plt.title('Actual vs predict', fontdict={'size': 20})
    # ax_x_vs_y(fea_x, target, prop,
    #                   show_xticks=True, show_yticks=True,
    #                   show_xlabel=True, show_ylabel=True,
    #                   show_mae=True, show_model=model)
    #
    # # cmap_values = [abs(fea_x[i] - target[i]) for i in range(len(fea_x))]
    # # vmin, vmax = 0, max(cmap_values) + 0.01
    # vmin, vmax = 0, 0.3
    #
    # cticks = (0, 0.1, 0.2, 0.3)
    # bar_label = '|DFT - target|' + r'$\/(\frac{eV}{atom})$'
    # bar_label_size = 18
    # bar_tick_size = 15
    # add_colorbar(fig,
    #              bar_label,
    #              cticks,
    #              # 'plasma_r',
    #              'coolwarm',
    #              vmin, vmax,
    #              [0.91, 0.2, 0.015, 0.6],
    #              bar_label_size, bar_tick_size, 1, 2)
    #
    plt.show()
    # savename = 'Fig_tgt_target_mae%s' % EXT
    # fig.savefig(os.path.join(FIG_DIR, savename))
    # plt.close()


def heatmapplotter(df, cmap, title='Formation energy of 212-MABs', label='formation energy (eV/atom)',
                   png_name='Ef_heatmap',
                   png_dir=r'G:\high-throughput-workflow\allthrough_heatmap\fomation_energy_heatmap',
                   conner_label=True,
                   conner_label_mask=None,
                   ):
    ### heatmap plotter ---------------------------------------------------
    params = yqsun_plot_font.__default_params__
    params.update({'font.size': '18', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)
    ticks_size = 17
    title_size = 24
    EXT = '.jpg'
    figsize = (8, 6)
    dpi = 800
    # plt.rcParams["xtick.direction"] = 'in'
    # plt.rcParams["ytick.direction"] = 'in'
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title, size=title_size, weight='bold')
    # print(df)
    values = df.to_numpy().flatten().tolist()
    # values = values.remove(1.0)
    values = sorted(values, reverse=True)
    values = [i for i in values if i != 1.0]
    # print(values)
    vmax = max(values)
    sns.heatmap(data=df,
                vmin=0,  # 图例（右侧颜色条color bar）中最小显示值
                vmax=vmax,  # 图例（右侧颜色条color bar）中最大显示值
                cmap=plt.get_cmap(cmap),  # 使用matplotlib中的颜色盘 'Set3', 'tab20c', 'Greens', 'Greens_r',
                # center=0.4,  # color bar的中心数据值大小，可以控制整个热图的颜盘深浅
                annot=True,  # 默认为False，当为True时，在每个格子写入data中数据
                fmt=".2f",  # 设置每个格子中数据的格式，参考之前的文章，此处保留两位小数
                annot_kws={'size': 8, 'weight': 'bold'},  # 设置格子中数据的大小、粗细、颜色
                # linewidths=0.8,  # 每个格子边框宽度，默认为0
                # linecolor='black',  # 每个格子边框颜色,默认为白色
                # cbar=False,  # 右侧图例(color bar)开关，默认为True显示
                cbar=True,  # 右侧图例(color bar)开关，默认为True显示
                cbar_kws={
                    'label': label,  # color bar的名称
                    # 'shrink': 0.8,
                    # 'orientation': 'vertical',  # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                    'orientation': 'horizontal',  # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                    "ticks": np.arange(0, vmax + 0.1, 0.05),  # color bar中刻度值范围和间隔
                    "format": "%.2f",  # 格式化输出color bar中刻度值
                    "pad": 0.205,  # color bar与热图之间距离，距离变大热图会被压缩
                },
                # mask=df == 1.0,  # 热图中显示部分数据：显示数值小于6的数据
                # xticklabels=['三连啊', '关注公众号啊', 'pythonic生物人', '收藏啊', '点赞啊', '老铁三连三连'],
                # # x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
                yticklabels=True,  # y轴方向刻度标签开关、同x轴
                xticklabels=True,  # y轴方向刻度标签开关、同x轴
                # yticklabels=False,  # y轴方向刻度标签开关、同x轴
                # xticklabels=False,  # y轴方向刻度标签开关、同x轴
                conner_label=conner_label,
                conner_label_mask=conner_label_mask,
                )

    ###
    plt.xticks(size=ticks_size, rotation=90)
    plt.yticks(size=ticks_size, rotation=0)
    plt.tick_params(labelsize=ticks_size, length=2, width=1)
    plt.subplots_adjust(top=0.922,
                        bottom=-0.03,
                        left=0.174,
                        right=0.951,
                        hspace=0.2,
                        wspace=0.2
                        )
    # print(plt.rcParams)
    # plt.show()
    # plt.savefig(r'G:\high-throughput-workflow\allthrough_heatmap\hull_heatmap_%s' % str(cmaps.index(cmap)))
    fig.savefig(png_dir + r'/' + png_name + '_%s' % cmap + EXT)
    plt.close()


class yqsun_plot_font():
    __default_params__ = {
        'font.family': 'sans-serif',
        # 'font.sans-serif': 'Helvetica',
        'font.sans-serif': 'Arial',
        'font.style': 'normal',
        'font.weight': 'normal',  # or 'blod'
        'font.size': '6',  # or large,small
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self.__default_params__)
        self.__dict__.update(kwargs)

    def set(self, kwargs):
        self.__dict__.update(kwargs)


def n_feas_error_plotter(hls, type, n_feas, n_plot, fig_name, main_label, set):
    if main_label:
        main_label_dir = main_label + '_figures'
    else:
        main_label_dir = 'figures'

    FIG_DIR = os.path.join('plotter', main_label_dir)
    EXT = '.jpg'
    title_size = 24
    title_pad = 20
    ticks_size = 17
    label_size = 22
    legend_size = 12
    num1, num2, num3, num4 = 1.00, 0.43, 3, 0
    # font params as dict
    params = yqsun_plot_font.__default_params__
    params.update({'font.size': '18', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)

    # set figure and axes
    fig, ax = plt.subplots(figsize=(8, 6), dpi=800)  # Create a figure and an axes.

    # set figure and axes
    # NUM_COLORS = len(self.A_plan_list)
    # cm = plt.get_cmap('gist_rainbow')
    # cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
    # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    # ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    markers = ["$♠$", "$♣$", "$♥$", "$♦$", ".", "o", "s", "P", "X", "*", "p", "D"]
    ax.set_prop_cycle(
        color=[(0, 0.4, 0.8), (0.4, 0.8, 0), (0.5, 0, 0.8), (0.9, 0.1, 0), (0.9, 0.5, 0), 'c', (0.8, 0.75, 0), 'k', 'b',
               'g', 'r', 'y'])
    # plt.xticks(np.arange(len(self.rule.keys())), self.rule.keys())
    # for ele, m in zip(self.A_plan_list, markers):

    from data.data import ml_out

    ml_results = ml_out(main_label)

    from check_log import md_labels
    labels = md_labels(
        layers_list=hls)
    # print(labels)

    # for n in sorted(opt_params.values()):
    # to_plot = {}
    y_ticks = []
    for label in labels:
        if n_plot == None:
            # x = sorted([i for i in range(100, 1100, 100)] + [i for i in range(10, 210, 10)])
            x = sorted([i for i in range(10, n_feas + 10, 10)])
        else:
            x = sorted(n_plot)

        if type == 'mae':
            y_label = 'mae'
        if type == 'rmse':
            y_label = 'rmse'

        module_label = label + "_layer_1000_relu"
        if set in ['all', 'train', 'test', 'validation', 'val_test']:
            y = [ml_results[str(n) + '_' + module_label][set][y_label] for n in x]

        y_ticks.extend(y)
        # for n in x:
        #     best_module_out = ml_results[str(n) + '_' + best_module_label]['test']
        #     print(best_module_out)
        # ax.scatter(latex_list, df['formation_energy'], color='b', linestyle='-', marker='+')
        ax.scatter(x, y, linestyle='--', linewidths=1.4, marker=".", s=30, label=label)
        ax.plot(x, y, linestyle='--', linewidth=1.4, alpha=0.6)

    # ax.plot(transition_ele_list, df[self.prop], label=ele, linestyle='-', linewidth=0.5, marker="$♦$")
    #
    #     x = plot_dict.keys()
    #     y = plot_dict.values()
    #     print(x)
    #     print(y)
    #     ax.scatter(x, y, label=ele, marker="$♦$")
    #
    # # ax.set_ylim(0.15, 0.5)
    # ax.set_ylim(0.8, 8)
    # # ax.set_ylim(0, 3.5)
    # # ax.set_ylim(0, 30)

    if set:
        if set == 'test':
            title_set = 'Testing'
        if set == 'validation':
            title_set = 'Validation'
        if set == 'train':
            title_set = 'Training'
        if set == 'all':
            title_set = 'All'
    y_max = max(y_ticks)
    y_min = min(y_ticks)
    # print(y_max)
    # print(y_min)
    y_tick_max = ceil(y_max * 100) * 10
    y_tick_min = floor(y_min * 100) * 10
    # print(y_tick_max)
    # print(y_tick_min)
    delta_max = abs(y_tick_max - y_max * 1000)
    delta_min = abs(y_tick_min - y_min * 1000)
    if delta_max > 5:
        y_tick_max = y_tick_max
    else:
        if 3 < delta_max < 5:
            y_tick_max = y_tick_max + 5
        if delta_max < 3:
            y_tick_max = y_tick_max + 5
    if delta_min > 5:
        y_tick_min = y_tick_min + 5
    else:
        y_tick_min = y_tick_min

    plt.yticks(ticks=[0.001 * i for i in range(y_tick_min, y_tick_max, 5)],
               # size=tick_size,
               # rotation=0
               )
    if type == 'rmse':
        ax.set_ylabel('RMSE' + r'$\/(\frac{eV}{atom})$', fontsize=label_size,
                      weight='bold')  # Add a y-label to the axes.

        ax.set_title('%s sets RMSE vs feature number' % title_set,
                     fontsize=title_size, weight='bold', pad=title_pad)  # Add a title to the axes.

    if type == 'mae':
        ax.set_ylabel('MAE' + r'$\/(\frac{eV}{atom})$', fontsize=label_size,
                      weight='bold')  # Add a y-label to the axes.
        ax.set_title('%s sets MAE vs feature number' % title_set,
                     fontsize=title_size, weight='bold', pad=title_pad)  # Add a title to the axes.

    ax.set_xlabel('Feature number', fontsize=label_size, weight='bold')  # Add a y-label to the axes.
    # # plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xticks(ticks=x
               # , size=tick_size,
               # rotation=0
               )

    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)

    ax.grid(True, alpha=1, linestyle='--')
    # ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0, fontsize=11,  title="hidden layers")  # Set the site of legend
    # ax.legend(loc="upper right", title="hidden layers", fontsize=8)  # Set the site of legend

    ax.legend(bbox_to_anchor=(num1, num2), loc=num3, title="Hidden layers",
              fontsize=legend_size)  # Set the site of legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.88,
                        bottom=0.13,
                        left=0.15,
                        right=0.76,
                        hspace=0.2,
                        wspace=0.2)  # compress the right of the figure

    # plt.show()
    # fig.savefig()

    # savename = 'n_feas_accuracy.jpg' % EXT
    savename = fig_name + '%s' % EXT

    fig.savefig(os.path.join(FIG_DIR, savename))
    # fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()
    # fig.savefig(self.figname, bbox_inches='tight')  # tight makes legend complete show


def rename_feas(n=10):
    df = pd.read_csv(r'features\opt_nmis\optimal_features_cross_nmi.csv', header=0)
    df = df.set_index('Unnamed: 0', drop=True)
    from utils.utils import read_json

    _json = read_json(r'features\opt_features\opt_feas_colums.json')
    feas = _json["all_%s_optimal_features" % str(n)]

    df = df.loc[:, feas]
    df = df.loc[feas, :]
    # print(df.columns)
    # print(df.index)

    column = []
    for nn in df.columns:
        _class, _name = nn.split('|')[0], nn.split('|')[1]
        # print(_class, _name)
        # if str(nn).startswith('K') or str(nn).startswith('k'):
        #     _label = r'$\kappa_{%s}$' % nn[1:]
        # elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith('m'):
        #     _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
        # elif str(nn) == 'a_b':
        #     _label = 'a/b'
        # elif str(nn) == 'NC.1':
        #     nn = "NCo"
        #     _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        # else:
        #     _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        # column.append(_label)


def cross_nmi_plotter(n, feas_name_dict, label=None, is_plot=False):
    if label:
        opt_nmi_dir = label + '_opt_nmis'
        label_dir = label + '_figures'
        opt_feas_dir = label + '_opt_features'
    else:
        opt_nmi_dir = 'opt_nmis'
        label_dir = 'figures'
        opt_feas_dir = 'opt_features'

    # print(opt_nmi_dir)
    # print(label_dir)
    # print(opt_feas_dir)

    cross_nmi_path = os.path.join(r'features', opt_nmi_dir, 'cross_nmi.csv')
    df = pd.read_csv(cross_nmi_path, header=0)
    df = df.set_index('Unnamed: 0', drop=True)
    opt_feas_colums = os.path.join(r'features', opt_feas_dir, 'opt_feas_colums.json')
    _json = read_json(opt_feas_colums)
    feas = _json["all_%s_optimal_features" % str(n)]

    df = df.loc[:, feas]
    df = df.loc[feas, :]
    full_name_column = df.columns
    full_name_index = df.index

    df.columns = [feas_name_dict[val]["simple"] for val in full_name_column]
    df.index = [feas_name_dict[val]["simple"] for val in full_name_index]
    # print(df.columns)
    # print(df.index)

    if is_plot == True:
        # cmaps = ['Greens']
        cmaps = ['Blues']
        # cmaps = ['co']
        # cmaps = ['coolwarm']
        savename = str(n) + '_cross_nmi'
        # savename = str(n) + 'test_cross_nmi'
        FIG_DIR = os.path.join('plotter', label_dir)
        for cmap in cmaps:
            heatmapplotter(df, cmap, title='Cross normalized mutual information', label='',
                           png_name=savename,
                           png_dir=FIG_DIR,
                           conner_label=False,
                           conner_label_mask=None,
                           )


def target_nmi_plotter(n, feas_name_dict, label, is_plot=False):
    # font params as dict
    params = yqsun_plot_font.__default_params__
    params.update({'font.size': '20', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)
    title_size = 21
    label_size = 22
    ticks_size = 20
    figsize = (8, 6)
    dpi = 800
    EXT = '.jpg'
    if label:
        opt_nmi_dir = label + '_opt_nmis'
        label_dir = label + '_figures'
        opt_feas_dir = label + '_opt_features'
    else:
        opt_nmi_dir = 'opt_nmis'
        label_dir = 'figures'
        opt_feas_dir = 'opt_features'

    # print(opt_nmi_dir)
    # print(label_dir)
    # print(opt_feas_dir)

    # fig_name = r'tgt_nmi'
    fig_name = r'tgt_nmi'
    FIG_DIR = os.path.join('plotter', label_dir)

    opt_feas_colums = os.path.join(r'features', opt_feas_dir, 'opt_feas_colums.json')
    _json = read_json(opt_feas_colums)
    feas = _json["all_%s_optimal_features" % str(n)]
    target_nmi_path = os.path.join(r'features', opt_nmi_dir, 'target_nmi.csv')
    data = pd.read_csv(target_nmi_path)
    data = data.set_index(data.columns[0])
    data.index.name = 'feature'
    data = data.loc[feas, :]
    data['simple'] = [feas_name_dict[val]["simple"] for val in feas]
    data.columns = ['score', 'simple']
    df = data.sort_values(by="score", ascending=False)
    cms = [cm.Blues(float(c) / len(df)) for c in [15] * len(df)]
    # print(df)
    df = df.sort_values(by='score', ascending=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title('Normalized mutual information with target', size=title_size, weight='bold')
    plt.barh(list(range(0, 4 * n, 4)), df['score'], height=3.7, linewidth=1, color=cms, alpha=1)
    plt.xticks(size=ticks_size)
    plt.yticks(list(range(0, 4 * n, 4)), df['simple'], size=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)
    # plt.tick_params(direction='out', length=1.2, width=0.5, size=6)
    plt.ylabel('Features', size=label_size, weight='bold')
    plt.xlabel('Target NMI', size=label_size, weight='bold')
    plt.subplots_adjust(top=0.922,
                        bottom=0.13,
                        left=0.22,
                        right=0.91,
                        hspace=0.2,
                        wspace=0.2
                        )
    # plt.show()
    savename = str(n) + '_' + fig_name + '%s' % EXT
    fig.savefig(os.path.join(FIG_DIR, savename), bbox_inched="tight")
    plt.close()


def feature_rank_plotter(n, feas_name_dict, label, is_plot=False):
    # font params as dict
    params = yqsun_plot_font.__default_params__
    # # set params to rcParams (control the font)
    params.update({'font.size': '20', 'font.weight': 'bold'})
    rcParams.update(params)
    title_size = 24
    label_size = 22
    ticks_size = 20
    figsize = (8, 6)
    dpi = 800
    EXT = '.jpg'
    if label:
        opt_nmi_dir = label + '_opt_nmis'
        label_dir = label + '_figures'
        opt_feas_dir = label + '_opt_features'
    else:
        opt_nmi_dir = 'opt_nmis'
        label_dir = 'figures'
        opt_feas_dir = 'opt_features'

    # print(opt_nmi_dir)
    # print(label_dir)
    # print(opt_feas_dir)

    # y = list(range(0, n*4, 4))
    y = [i for i in range(4, n * 4 + 4, 4)]
    rank = [i / 10 for i in y]
    # rank = [i - 1 for i in rank]
    fig_name = r'feature_rank'
    FIG_DIR = os.path.join('plotter', label_dir)

    opt_feas_colums = os.path.join(r'features', opt_feas_dir, 'opt_feas_colums.json')

    _json = read_json(opt_feas_colums)
    feas = _json["all_%s_optimal_features" % str(n)]

    cms = [cm.nipy_spectral(float(c) / len(y)) for c in range(len(y))]
    # cms = [cm.viridis(float(c) / len(df)) for c in range(len(df))]
    # cms = [cm.cividis(float(c) / len(df)) for c in range(len(df))]
    # cms = [cm.cool(float(c) / len(df)) for c in range(len(df))]
    # cms = [cm.Blues(float(c) / len(df)) for c in range(len(df))]
    # print(cms)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title('Features ranked by MOD-selection', size=title_size, weight='bold')
    plt.barh(y, rank, height=3.7, linewidth=1, color=cms, alpha=0.6)
    plt.yticks(ticks=y, labels=reversed(list([feas_name_dict[i]['simple'] for i in feas])), size=ticks_size)
    plt.xticks([])
    # plt.xticks()
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)
    plt.ylabel('Features', size=label_size, weight='bold')
    plt.xlabel('Rank', size=label_size, weight='bold')

    plt.subplots_adjust(top=0.922,
                        bottom=0.13,
                        left=0.22,
                        right=0.91,
                        hspace=0.2,
                        wspace=0.2
                        )
    # plt.show()
    savename = str(n) + '_' + fig_name + '%s' % EXT
    fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()


def confusion_matrix_plotter(training_prop, thresh=0.1, fea_num=20, hl='',
                             module_label=None, space_group=None, label=None, set=None):
    params = yqsun_plot_font.__default_params__
    # print(params)
    params.update({'font.size': '20', 'font.weight': 'bold'})
    # # set params to rcParams (control the font)
    rcParams.update(params)
    EXT = '.jpg'

    title_font = 24
    pad = 20
    text_font = 18
    text_size = 18
    ticks_size = 20
    label_size = 22
    s = 30

    # print(ticks_size)
    if label:
        label_dir = label + '_figures'
    else:
        label_dir = 'figures'

    FIG_DIR = os.path.join('plotter', label_dir)

    fig = plt.figure(figsize=(8, 6), dpi=800)
    # print(ticks_size)
    ax_class(
        label,
        title_font,
        pad,
        text_font,
        text_size,
        ticks_size,
        s,
        label_size,
        True, True, True, thresh, fea_num, hl, module_label, space_group, set)
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    for i in spines_dict:
        tk.spines[i].set_linewidth(1.5)
    plt.subplots_adjust(top=0.90,
                        bottom=0.17,
                        left=0.15,
                        right=0.85,
                        hspace=0.2,
                        wspace=0.2)
    # plt.tight_layout()
    # plt.show()

    if thresh == 0:
        savename = 'Ef%s' % EXT if training_prop == 'Ef' else 'Ed%s' % EXT
    else:
        savename = 'Ef_%s_%s_%s' % (
            str(int(1000 * thresh)), module_label, EXT) if training_prop == 'Ef' else 'Ed_%s_%s_%s' % (
            str(int(1000 * thresh)), module_label, EXT)
    if set:
        savename = 'no_' + set + '_' + savename
    fig.savefig(os.path.join(FIG_DIR, savename))
    # fig.savefig(os.path.join(FIG_DIR, 'test' + savename))
    plt.close()


def ax_class(
        label,
        title_font,
        pad,
        text_font,
        text_size,
        ticks_size,
        s,
        label_size, show_xlabel, show_ylabel, show_model, thresh=0.1, fea_num=20, hl='',
        module_label='', space_group=None, set=None):
    prop = 'Ed'
    # print(space_group)
    if set:
        # print('using %s set data' % set)
        set_id_json = read_json(
            os.path.join(r'data\data', '%s_%s_set_id.json' % (label, module_label)))
        actual = [set_id_json[id]['hull_distance'] for id in set_id_json if set_id_json[id]['set'] != set]
        pred = [set_id_json[id]['predict'] for id in set_id_json if set_id_json[id]['set'] != set]
    else:
        compounds = get_compounds(space_group)  ## ids
        actual = get_actual(prop, compounds, space_group)
        pred = get_pred(prop, label, module_label)

    # print(len(pred))
    # print(len(actual))
    xticks_Ed = (-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    yticks_Ed = (-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    xlim_Ed, ylim_Ed = (-0.1, 0.85), (-0.1, 0.85)

    c_stable = tableau_colors()['purple']
    c_unstable = tableau_colors()['orange']
    plt.plot([thresh, thresh], ylim_Ed, ls='--', color='black', alpha=0.4)
    plt.plot(xlim_Ed, [thresh, thresh], ls='--', color='black', alpha=0.4)
    fea_num = str(fea_num)
    labels = []
    # print(len(actual))
    for i in range(len(actual)):
        if (actual[i] <= thresh) and (pred[i] <= thresh):
            label = 'tp'
        elif pred[i] <= thresh:
            # print(pred[i])
            # print(thresh)
            label = 'fp'
        elif (actual[i] > thresh) and (pred[i] > thresh):
            label = 'tn'
        else:
            label = 'fn'
        labels.append(label)

    data = {label: {'actual': [actual[i] for i in range(len(labels)) if labels[i] == label],
                    'pred': [pred[i] for i in range(len(labels)) if labels[i] == label]}
            for label in ['tp', 'fp', 'tn', 'fn']}

    tp = len(data['tp']['actual'])
    fp = len(data['fp']['actual'])
    tn = len(data['tn']['actual'])
    fn = len(data['fn']['actual'])

    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)
    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    # print(tp)
    # print(tn)
    # print(fp)
    # print(fn)

    acc = (tp + tn) / (tp + tn + fp + fn)
    if fp + tn == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    alpha = 0.95

    for label in data:
        if label == 'tp':
            color = c_stable
            edgecolor = c_stable
            marker = 'o'
        elif label == 'tn':
            color = c_stable
            edgecolor = c_stable
            marker = 'o'
        elif label == 'fp':
            color = c_unstable
            edgecolor = c_unstable
            marker = 'o'
        elif label == 'fn':
            color = c_unstable
            edgecolor = c_unstable
            marker = 'o'
        pred = data[label]['pred']
        actual = data[label]['actual']
        # print(ticks_size)
        ax_generic_scatter(pred, actual,
                           colors=color,
                           marker=marker,
                           s=s,
                           edgecolors=edgecolor,
                           xticks=(True, xticks_Ed), yticks=(True, yticks_Ed), ticks_size=ticks_size,
                           xlim=xlim_Ed, ylim=ylim_Ed, label_size=label_size, alpha=alpha, diag=False)

    plt.text(-0.075, -0.09,
             'TP = %i' % (tp),
             color=c_stable,
             fontsize=text_font,
             verticalalignment='bottom')
    plt.text(-0.075, 0.83,
             'FP = %i' % (fp),
             color=c_unstable,
             verticalalignment='top',
             fontsize=text_font)
    plt.text(0.82, 0.83,
             'TN = %i' % (tn),
             color=c_stable,
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=text_font)
    plt.text(0.82, -0.09,
             'FN = %i' % (fn),
             color=c_unstable,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=text_font)
    plt.tick_params(direction='out', length=1.2, width=0.5)
    plt.title('Confusion matrix', fontsize=title_font, weight='bold', pad=pad)

    if show_xlabel:
        plt.xlabel(r'$\Delta$' + r'$\it{H}$' + r'$_{d, prediction}$' + r'$\/(\frac{eV}{atom})$', weight='bold')
    else:
        plt.xlabel('')
        plt.gca().xaxis.set_ticklabels([])
    if show_ylabel:
        plt.ylabel(r'$\Delta$' + r'$\it{H}$' + r'$_{d, DFT}$' + r'$\/(\frac{eV}{atom})$', weight='bold')
    else:
        plt.ylabel('')
        plt.gca().yaxis.set_ticklabels([])

    if show_model:
        xpos, ypos = -0.08, 0.48
        text = "Feature number : %s\n" % fea_num + "Hidden Layers : %s\n" % hl + "Accuracy : %.3f\nFPR : %.3f\n" % (
        acc, fpr)
        # "Precision : %.3f\nRecall : %.3f\nF1 : %.3f\n" % (prec, rec ,f1)
        plt.text(xpos, ypos,
                 text,
                 fontsize=text_size,
                 horizontalalignment='left',
                 verticalalignment='bottom', weight='bold')


def get_best_n_target_nmi_feas(n=30):
    _json = read_json(r'features\opt_features\opt_feas_colums.json')
    feas = _json["all_%s_optimal_features" % str(n)]
    # print(feas)
    data = pd.read_csv(r'features\opt_nmis\optimal_features_target_nmi_.csv')
    data = data.set_index(data.columns[0])
    data.index.name = 'feature'
    data = data.loc[feas, :]
    # data['simple'] = simple_name
    # data.columns = ['score', 'simple']
    data.columns = ['score']
    df = data.sort_values(by="score", ascending=False)
    return df


class feature_simple_pretty_name():
    def __init__(self, opt_num, fmt):
        opt_feas_dir = r'features\opt_features'
        name = 'all_' + str(opt_num) + '_optimal_features.' + fmt
        opt_feas = os.path.join(opt_feas_dir, name)
        if fmt == 'xlsx':
            opt_feas_df = pd.read_excel(opt_feas)
        if fmt == 'csv':
            opt_feas_df = pd.read_csv(opt_feas)
        opt_feas_df = opt_feas_df.set_index('id')
        opt_feas_df = opt_feas_df.iloc[:, :-1]
        feas_name = opt_feas_df.columns.values
        # print(feas_name)
        feas_name_dict = {
            'ElementProperty|MagpieData mean MendeleevNumber': {'simple': 'meanMN', 'pretty': 'mean MendeleevNumber'},
            'ElementProperty|MagpieData maximum NfValence': {'simple': 'maxNf', 'pretty': 'maximum NfValence'},
            'StructuralHeterogeneity|max relative bond length': {'simple': 'maxBL',
                                                                 'pretty': 'max relative bond length'},
            'ElementProperty|MagpieData avg_dev NsValence': {'simple': 'adNs', 'pretty': 'avg_dev NsValenc'},
            'StructuralHeterogeneity|range neighbor distance variation': {'simple': 'rNDV',
                                                                          'pretty': 'range neighbor distance variation'},
            'ElementProperty|MagpieData mean NpUnfilled' 'AtomicOrbitals|gap_AO': {'simple': 'meanNpU',
                                                                                   'pretty': 'mean NpUnfilled'},
            'ChemicalOrdering|mean ordering parameter shell 1': {'simple': 'meanOPS1',
                                                                 'pretty': 'mean ordering parameter shell 1'},
            'StructuralHeterogeneity|min relative bond length': {'simple': 'minBL',
                                                                 'pretty': 'min relative bond length'},
            'ElementProperty|MagpieData avg_dev AtomicWeight': {'simple': 'adAW', 'pretty': 'avg_dev AtomicWeight'},
            'ChemicalOrdering|mean ordering parameter shell 2': {'simple': 'meanOPS2',
                                                                 'pretty': 'mean ordering parameter shell 2'},
            'ElementProperty|MagpieData mode NUnfilled': {'simple': 'NU', 'pretty': 'NUnfilled'},
            'EwaldEnergy|ewald_energy_per_atom': {'simple': 'EE', 'pretty': 'Ewald energy per atom'},
            'AtomicOrbitals|LUMO_element': {'simple': 'LUMO', 'pretty': 'LUMO element'},
            'StructuralHeterogeneity|maximum neighbor distance variation': {'simple': 'maxNDV',
                                                                            'pretty': 'maximum neighbor distance variation'},
            'ElementProperty|MagpieData range NUnfilled': {'simple': 'rNU', 'pretty': 'range NUnfilled'},
            'StructuralHeterogeneity|mean neighbor distance variation': {'simple': 'meanNDV',
                                                                         'pretty': 'mean neighbor distance variation'},
            'ElementProperty|MagpieData maximum NUnfilled': {'simple': 'maxNU', 'pretty': 'maximum NUnfilled'},
            'ElementProperty|MagpieData mean GSvolume_pa': {'simple': 'meanV', 'pretty': 'mean volume'},
            'ElementProperty|MagpieData mean Electronegativity': {'simple': 'meanEn',
                                                                  'pretty': 'mean Electronegativity'},
            'ChemicalOrdering|mean ordering parameter shell 3': {'simple': 'meanOPS3',
                                                                 'pretty': 'mean ordering parameter shell 3'},
            'ValenceOrbital|frac s valence electrons': {'simple': 'Fs', 'pretty': 's valence electrons fraction'},
            'YangSolidSolution|Yang delta': {'simple': 'YD', 'pretty': 'Yang delta'},
            'ElementProperty|MagpieData range NValence': {'simple': 'rNV', 'pretty': 'range NValence'},
            'StructuralHeterogeneity|avg_dev neighbor distance variation': {'simple': 'adNDV',
                                                                            'pretty': 'avg_dev neighbor distance variation'},
            'ElementProperty|MagpieData mean NUnfilled': {'simple': 'meanNU', 'pretty': 'mean NUnfilled'},
            'ElementProperty|MagpieData maximum Electronegativity': {'simple': 'maxEn',
                                                                     'pretty': 'maximum Electronegativity'},
            'ElementProperty|MagpieData mean CovalentRadius': {'simple': 'meanCR', 'pretty': 'mean CovalentRadius'},
            'ValenceOrbital|avg d valence electrons': {'simple': 'adVE', 'pretty': 'avg d valence electrons'},
            'StructuralHeterogeneity|mean absolute deviation in relative cell size': {'simple': 'madRCS',
                                                                                      'pretty': 'mean absolute deviation in relative cell size'},
            'ValenceOrbital|frac f valence electrons': {'simple': 'Ff', 'pretty': 'f valence electrons fraction'},
            'ElementProperty|MagpieData minimum Electronegativity': {'simple': 'minEn',
                                                                     'pretty': 'minimum Electronegativity'},
            'ElementProperty|MagpieData range GSvolume_pa': {'simple': 'rV', 'pretty': 'range volume'},
            'AtomicOrbitals|LUMO_energy': {'simple': 'LUMOe', 'pretty': 'LUMO energy'},
            'ElementProperty|MagpieData avg_dev MendeleevNumber': {'simple': 'adMN',
                                                                   'pretty': 'avg_dev MendeleevNumber'},
            'ElementProperty|MagpieData avg_dev Electronegativity': {'simple': 'adEn',
                                                                     'pretty': 'avg_dev Electronegativity'},
            'ElementProperty|MagpieData mean Number': {'simple': 'meanN', 'pretty': 'mean Number'},
            'ElementProperty|MagpieData avg_dev Number': {'simple': 'adN', 'pretty': 'avg_dev Number'},
            'ElementProperty|MagpieData minimum NUnfilled': {'simple': 'minNU', 'pretty': 'minimum NUnfilled'},
            'ElementProperty|MagpieData avg_dev NValence': {'simple': 'adNV', 'pretty': 'avg_dev NValence'}
        }
        write_json(feas_name_dict, 'figures/hex_orth_comp_struc_' + str(opt_num) + '_opt_feas_name.json')


if __name__ == '__main__':
    from utils.utils import check_dir_existence
    import pprint

    ### run renamer.py

    ### make opt_feas_name.json in plotter dir
    make = False
    # make = True
    is_plot = True

    main_label = 'orth'
    # main_label = 'hex'
    # main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    if main_label:
        opt_nmi_dir = main_label + '_opt_nmis'
        main_label_dir = main_label + '_figures'
        opt_feas_dir = main_label + '_opt_features'
        if main_label == 'orth':
            opt_num = 15
        if main_label == 'hex':
            opt_num = 10
        if main_label == 'hex_orth':
            opt_num = 10
        if main_label == 'hex_orth_space':
            opt_num = 15

    else:
        opt_nmi_dir = 'opt_nmis'
        main_label_dir = 'figures'
        opt_feas_dir = 'opt_features'
    check_dir_existence(os.path.join(r'plotter', main_label_dir))
    if make == True:
        # figures_dir = os.path.join(r'plotter', main_label_dir)
        # check_dir_existence(figures_dir)
        # fea_name_dict = read_json(os.path.join(figures_dir, r'opt_feas_name.json'))
        # fea_names = feature_simple_pretty_name(40, 'csv')
        # fea_name_dict = read_json('figures/hex_orth_comp_struc_40_opt_feas_name.json')
        # fea_name_dict = read_json('figures/hex_orth_comp_struc_40_opt_feas_name.json')
        # fea_name_dict = read_json('old_hex_orth_space_figures/opt_feas_name.json')
        # fea_name_dict = read_json('old_orth_figures/opt_feas_name.json')
        # fea_name_dict = read_json('old_hex_figures/opt_feas_name.json')
        try:
            fea_name_dict = read_json(main_label + '_figures/old_opt_feas_name.json')
        except:
            fea_name_dict = read_json(main_label + '_figures/opt_feas_name.json')

        opt_feas_csv = os.path.join(r'features', opt_feas_dir,
                                    'all_' + str(opt_num) + '_optimal_features.csv')
        opt_feas = pd.read_csv(opt_feas_csv)
        best_fea_group_list = opt_feas.columns.values[1:][:-1]
        best_feas_dict = {i: {"simple": "", "pretty": ""} for i in best_fea_group_list}
        for i in best_feas_dict:
            if i in fea_name_dict.keys():
                best_feas_dict[i] = fea_name_dict[i]

        write_json(best_feas_dict,
                   os.path.join(r'plotter', main_label_dir, 'opt_feas_name.json'))

    if is_plot == True:
        fea_name_dict = read_json(os.path.join(r'plotter', main_label_dir, 'opt_feas_name.json'))
        cross_nmi_plotter(opt_num, fea_name_dict, main_label, True)
        target_nmi_plotter(opt_num, fea_name_dict, main_label, True)
        feature_rank_plotter(opt_num, fea_name_dict, main_label, True)
