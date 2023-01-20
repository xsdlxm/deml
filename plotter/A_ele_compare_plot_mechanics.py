import numpy as np
import pandas as pd
from pymatgen.io.vasp.inputs import Structure
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
import json
import os


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files


def get_gibbs_df(path):
    df = pd.read_excel(path, header=0)
    # print(df0)
    return df


def get_st_list(dir):
    files = file_name(dir)

    st_list = []

    for i in files:
        j = dir + '/' + i
        st_list.append(j)
    # st_list.pop()
    return st_list


def get_best_formula(formula):
    a = formula.split(' ')
    b = [1, 2, 3]
    for i in a:
        if '1' in i:
            b[1] = i
        if 'B2' in i:
            b[2] = i
        if '2' in i and 'B' not in i:
            b[0] = i
    return b[0] + b[1] + b[2]


class mechanic_plotter(object):

    def __init__(self, prop="hardness (GPa)", ylabel="hardness (GPa)", title='Vickers hardness', figname="mec_prop"):
        # A_plan_str = "Ga"
        # A_plan_str = "Ga, Tl"
        A_plan_str = "Ga, In, Tl, Si, Zn, Cd"
        self.rule = {'Sc': 0, 'Y': 1, 'Ti': 2, 'Zr': 3, 'Hf': 4, 'V': 5, 'Nb': 6, 'Ta': 7, 'Cr': 8, 'Mo': 9, 'W': 10,
                     'Mn': 11, 'Tc': 12, 'Fe': 13, 'Ru': 14, 'Co': 15, 'Rh': 16, 'Ni': 17}
        self.A_plan_list = A_plan_str.split(", ")
        self.ylabel = ylabel
        self.title = title
        self.prop = prop
        self.figname = figname
        self.ele_list_plotter()

    def ele_list_plotter(self):

        # font params as dict
        params = {'font.family': 'serif',
                  'font.serif': 'Times New Roman',
                  'font.style': 'normal',
                  'font.weight': 'normal',  # or 'blod'
                  'font.size': '14',  # or large,small
                  }
        # set params to rcParams (control the font)
        rcParams.update(params)

        # set figure and axes
        fig, ax = plt.subplots()  # Create a figure and an axes.

        # set figure and axes
        # NUM_COLORS = len(self.A_plan_list)
        # cm = plt.get_cmap('gist_rainbow')
        # cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
        # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        # ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        markers = ["$♠$", "$♣$", "$♥$", "$♦$", ".", "o", "s", "P", "X", "*", "p", "D"]
        ax.set_prop_cycle(
            color=[(0, 0.4, 0.8), (0.4, 0.8, 0), (0.5, 0, 0.8), (0.9, 0.1, 0), (0.9, 0.5, 0), (0.8, 0.75, 0), 'c', 'k',
                   'b', 'g', 'r', 'y'])
        # plt.xticks(np.arange(len(self.rule.keys())), self.rule.keys())
        # for ele, m in zip(self.A_plan_list, markers):
        for ele in self.A_plan_list:
            # read xls to df
            xls_dir = 'G:\high-throughput-workflow\liyu_lasted\query_elastic_properties_judge_stability\plot\\new_mechanic_props'
            path = xls_dir + '\\' + ele + '_mechanics_properties.xlsx'
            df = pd.read_excel(path, header=0)
            # print(df)

            # plot_dict = {'Sc': 0, 'Y': 0, 'Ti': 0, 'Zr': 0, 'Hf': 0, 'V': 0, 'Nb': 0, 'Ta': 0, 'Cr': 0, 'Mo': 0, 'W': 0,
            #     'Mn': 0, 'Tc': 0, 'Fe': 0, 'Ru': 0, 'Co': 0, 'Rh': 0, 'Ni': 0}
            plot_dict = {'Sc': 1, 'Y': 'NaN', 'Ti': 'NaN', 'Zr': 'NaN', 'Hf': 'NaN', 'V': 'NaN', 'Nb': 'NaN',
                         'Ta': 'NaN', 'Cr': 'NaN', 'Mo': 'NaN', 'W': 'NaN',
                         'Mn': 'NaN', 'Tc': 'NaN', 'Fe': 'NaN', 'Ru': 'NaN', 'Co': 'NaN', 'Rh': 'NaN', 'Ni': 'NaN'}
            # get best sorted formula and LaTex form
            element = df['element'].to_list()
            print(element)
            mec = df[self.prop].to_list()
            for i in element:
                plot_dict[i] = mec[element.index(i)]
            # ax.scatter(latex_list, df['formation_energy'], color='b', linestyle='-', marker='+')
            # ax.scatter(transition_ele_list, df['formation_energy'], linestyle='-', marker=m)
            # ax.plot(transition_ele_list, df[self.prop], label=ele, linestyle='-', linewidth=0.5, marker="$♦$")

            x = plot_dict.keys()
            y = plot_dict.values()
            print(x)
            print(y)
            ax.scatter(x, y, label=ele, marker="$♦$")

        # ax.set_ylim(0.15, 0.5)
        ax.set_ylim(0.8, 8)
        # ax.set_ylim(0, 3.5)
        # ax.set_ylim(0, 30)
        ax.set_ylabel(self.ylabel)  # Add a y-label to the axes.
        # plt.axhline(y=0.5, color='r', linestyle='--')
        # plt.xticks(size=9, rotation=-30)
        ax.set_title(self.title)  # Add a title to the axes.
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0, fontsize=11)  # Set the site of legend
        # fig.subplots_adjust(right=0.75) # compress the right of the figure
        # plt.savefig()
        fig.savefig(self.figname, bbox_inches='tight')  # tight makes legend complete show


if __name__ == '__main__':
    '''
        {0: {'formula': 'Mo2As1B2', 'C11': 479.386, 'C12': 186.60099, 'C13': 102.87976, 'C23': 182.76254, 
        'C22': 217.26922, 'C33': 393.32396, 'C44': 121.23447, 'C55': 126.37027, 'C66': 157.40188, 
        'k_vrh (GPa)': 216.977097860924, 'g_vrh (GPa)': 105.118762686865, 'E_mod (GPa)': 271.51016632361, 
        'poisson_ratio': 0.291444835268866, 'anisotropy': 2.02531151773943, 'k_g_ratio': 2.06411388713993, 
        'cauchy_pressure': '[61.528069999999985, -23.49051, 29.19910999999999]', 'gruneisen': 1.72090844295746, 
        "debye_temperature", "hardness (GPa)"
        'No.': 0}, 
    '''
    # M_ele_tongzu_plotter('element', 'debye_temperature', 'debye temperature (K)', 'Debye temperature')
    # A_M_ele_tongzu_plotter('element', 'mean_v', 'mean velocity (m/s)', 'Mean velocity')
    # mechanic_plotter("hardness (GPa)", "hardness (GPa)", 'Vickers hardness', "Hv")
    # mechanic_plotter('anisotropy', "Au", 'Universal anisotropy', "Au_min")
    # mechanic_plotter('anisotropy', "Au", 'Universal anisotropy', "Au_all")
    a = mechanic_plotter('k_g_ratio', 'K/G ratio', 'K/G ratio', "k_g_ratio")
