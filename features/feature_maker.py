from modnet.preprocessing import MODData
import pandas as pd
from pathlib import Path
from collections import defaultdict
import csv
import json
import os
import numpy as np
from utils.utils import read_json, write_json
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from matminer.featurizers.structure import (
    GlobalSymmetryFeatures,
    DensityFeatures)


def get_structure_paths(dir):
    structures_list = os.listdir(dir)
    paths = [os.path.join(dir, i) for i in structures_list]
    return paths


def excell_to_df(path):
    df = pd.read_excel(path, header=0)
    # print(df0)
    return df


def df_to_excell(path, df):
    if '.xlsx' in path:
        write = pd.ExcelWriter(path)
        df.to_excel(write)
        write.save()
        write.close()


class structures():
    def __init__(self, fmt='json', id_json_path='G:\codes\modnet\mab_ml\data\data\mab_form_comp_hull_spaceg.json',
                 structures_dir=r'G:\codes\modnet\mab_ml\data\mp_opted_structures', id_json_list=None,
                 structures_dir_list=None):
        self.fmt = fmt

        if id_json_list:
            self.id_json = {}
            for j in id_json_list:
                js = read_json(j)
                self.id_json.update(js)
        else:
            self.id_json = read_json(id_json_path)
        print(len(self.id_json))

        if structures_dir_list:
            self.structure_list = []
            for i in structures_dir_list:
                st_list = [os.path.join(i, vasp) for vasp in os.listdir(i)]
                self.structure_list.extend(st_list)
        else:
            self.structure_list = [os.path.join(structures_dir, vasp) for vasp in os.listdir(structures_dir)]

        self.get_sturctures_set()

    def __len__(self):
        return len(self.structure_list)

    def get_sturctures_set(self):
        structure_composition_union = {}
        if self.fmt == 'json':
            for id in self.id_json:
                _dict = self.id_json[id]
                space_group = _dict['space_group']
                vasp_dir = '../data/mp_opted_structures_' + space_group
                # 'Al(BMo)2_212_orth_mp.vasp'
                vasp_path = os.path.join(vasp_dir, _dict["pretty_formula"] + '_212_' + space_group + '_mp.vasp')
                crystal = Structure.from_file(vasp_path)
                crystal.add_oxidation_state_by_guess()
                _dict['structure'] = crystal
                _dict['target'] = _dict['hull_distance']
                structure_composition_union[id] = _dict
        # if self.fmt == 'poscar':
        #     for vasp in self.structure_list:
        #         crystal = Structure.from_file(vasp)
        #         crystal.add_oxidation_state_by_guess()
        #         formula = crystal.formula.replace(' ', '')
        #         _dict = self.id_json[formula]
        #         _dict['structure'] = crystal
        #         _dict['target'] = self.id_json[formula]['hull_distance']
        #         structure_composition_union[self.id_json[formula]['id']] = _dict

        return pd.DataFrame.from_dict(structure_composition_union).T


def read_json(file):
    with open(file, 'r') as f:
        json_comp = f.readline()
        json_to_dict = json.loads(s=json_comp)
    return json_to_dict


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def modify_ids(df):
    df_index_list = []
    if isinstance(df, pd.DataFrame):
        print('df')
        for idx in df.index:
            if '-' in idx:
                idx_ls = idx.split('-')
                # print(idx_ls)
                if idx_ls[0] == 'mp':
                    df_index_list.append('10' + idx_ls[-1])
                if idx_ls[0] == 'mvc':
                    df_index_list.append('11' + idx_ls[-1])
            else:
                df_index_list.append(idx)
        return df_index_list
    if isinstance(df, list):
        print('list')
        for idx in df:
            if '-' in idx:
                idx_ls = idx.split('-')
                if idx_ls[0] == 'mp':
                    df_index_list.append('10' + idx_ls[-1])
                if idx_ls[0] == 'mvc':
                    df_index_list.append('11' + idx_ls[-1])
            else:
                df_index_list.append(idx)
        return df_index_list


def get_features(df, feas_df, read_features=False, target_names=['hull_distance'],
                 featurizer=None, is_add_lattice=True, is_add_sym=True, is_opt=False, opt_feas_limit=50, n_jobs=14,
                 cross_nmi=None, target_nmi=None, use_precomputed_cross_nmi=False,
                 nmi_dir=r'G:\codes\modnet\mab_ml\features\opt_nmis',
                 feas_name='MAB_features', is_save_feas=True,
                 for_opt_params=True, opt_params=None,
                 opt_feas_name='optimal_features', is_save_opt_feas=True,
                 opt_dir=r'G:\codes\modnet\yqsun_test\features\opt_features',
                 fmt='csv',
                 ):
    from utils.utils import check_dir_existence

    check_dir_existence(nmi_dir)
    check_dir_existence(opt_dir)

    if isinstance(feas_df, pd.DataFrame) == True:
        read_features = True

    if read_features == True:
        print('feature number: %s' % (len(feas_df.columns) - 1))
        if 'id' in feas_df.columns.values:
            id = feas_df['id']
            feas_df = feas_df.drop('id', axis=1)

    data = MODData(materials=df['structure'],
                   targets=df['target'].values,
                   structure_ids=df.index,
                   target_names=target_names,
                   num_classes={i: 0 for i in target_names},
                   featurizer=featurizer,
                   df_featurized=feas_df,
                   )

    if read_features == False:
        data.featurize(n_jobs=n_jobs)

    # data.df_featurized.reset_index()
    df_feas = data.df_featurized
    print(df_feas)

    if is_add_lattice == True:
        constants = lattice_constant(df)
        constants = constants.set_index(df.index.values)
        # constants.index.name = 'id'
        # print(constants)
        df_feas = df_feas.join(constants)
        # df_feas = df_feas.merge(constants, 'outer' )

    if is_add_sym == True:
        ### need same id as df_feas
        # print(df_feas.index.values)
        gsf = spacegroup_system_featurizer(df)
        dsf = density_featurizer(df)

        gsf = gsf.set_index(df.index.values)
        dsf = dsf.set_index(df.index.values)

        df_feas = df_feas.join(gsf, rsuffix='r')
        df_feas = df_feas.join(dsf, rsuffix='r')

    # print(df_feas.iloc[:, -1])
    # print(df_feas.iloc[:, -4])
    # if 'Unnamed: 0' in df_feas.columns.values:
    #     df_feas = df_feas.drop(columns='Unnamed: 0')
    # df_feas.index = df.index
    # df_feas = get_no_zero_df(df_feas)
    # print(df_feas)

    if is_save_feas == True:
        df_feas = df_feas.set_index(df.index.values)
        df_feas.index.name = 'id'
        if fmt == 'xlsx':
            df_to_excell(feas_name + '.xlsx', df_feas)
        if fmt == 'csv':
            df_feas.to_csv(feas_name + '.csv')
        if fmt == 'both':
            df_to_excell(feas_name + '.xlsx', df_feas)
            df_feas.to_csv(feas_name + '.csv')

    if is_opt:
        if cross_nmi is not None:
            cross_nmi = pd.read_csv(cross_nmi)
            cross_nmi = cross_nmi.set_index(cross_nmi['Unnamed: 0'].values)
            cross_nmi = cross_nmi.drop(columns='Unnamed: 0')
            print(cross_nmi)

        if target_nmi is not None:
            target_nmi = pd.read_csv(target_nmi)
            target_nmi = target_nmi.set_index(target_nmi['Unnamed: 0'].values)
            target_nmi = target_nmi[target_names[0]]
            print(target_nmi)

        if for_opt_params == True:
            for opt_name, num in opt_params.items():
                data.feature_selection(n=num, is_write=True, cross_nmi=cross_nmi, target_nmi=target_nmi,
                                       use_precomputed_cross_nmi=use_precomputed_cross_nmi,
                                       save_dir=nmi_dir, n_jobs=n_jobs,
                                       label=opt_name)
                optimal_df = data.get_optimal_df()
                # print(opt_name)
                # print(optimal_df.columns.values)
                if read_features == True:
                    for target in target_names:
                        optimal_df[target] = df['target']
                if is_save_opt_feas == True:
                    if fmt == 'xlsx':
                        df_to_excell(os.path.join(opt_dir, opt_name + '.xlsx'), optimal_df)
                    if fmt == 'csv':
                        optimal_df.to_csv(os.path.join(opt_dir, opt_name + '.csv'))
                    if fmt == 'both':
                        df_to_excell(os.path.join(opt_dir, opt_name + '.xlsx'), optimal_df)
                        optimal_df.to_csv(os.path.join(opt_dir, opt_name + '.csv'))
        else:
            data.feature_selection(n=opt_feas_limit, cross_nmi=cross_nmi, target_nmi=target_nmi,
                                   use_precomputed_cross_nmi=use_precomputed_cross_nmi,
                                   save_dir=nmi_dir, n_jobs=n_jobs,
                                   label=opt_feas_name)
            optimal_df = data.get_optimal_df()
            optimal_df = optimal_df.set_index(df.index.values)
            # print(optimal_df.columns)
            if read_features == True:
                for target in target_names:
                    optimal_df[target] = df['target']
            if is_save_opt_feas == True:
                # name = os.path.join(save_dir, )
                if fmt == 'xlsx':
                    df_to_excell(os.path.join(opt_dir, opt_feas_name + '.xlsx'), optimal_df)
                if fmt == 'csv':
                    optimal_df.to_csv(os.path.join(opt_dir, opt_feas_name + '.csv'))
                if fmt == 'both':
                    df_to_excell(os.path.join(opt_dir, opt_feas_name + '.xlsx'), optimal_df)
                    optimal_df.to_csv(os.path.join(opt_dir, opt_feas_name + '.csv'))

    # optimal_descriptors = data.get_optimal_descriptors()
    # print(optimal_descriptors)
    # Creating modnetModel
    # from modnet.models import modnetModel
    # model = modnetModel([[['Hd']]], {'Hd': 1},
    #                     num_neurons=[[256], [64], [64], [32]],
    #                     )
    # model.fit(data)


def get_no_zero_df(df):

    return df


def lattice_constant(df):
    desired_features = {"a": [], "b": [], "c": []}
    fea_class = 'LatticeConstants|'
    for sg, s in zip(df['space_group'], df['structure']):
        lattice = s.lattice.as_dict()
        matrix = lattice['matrix']
        v1 = np.array(matrix[0])
        v2 = np.array(matrix[1])
        v3 = np.array(matrix[2])
        if sg == 'orth':
            v4 = v1 + v2
            v5 = v1 - v2
            norm1 = float(np.linalg.norm(v4))
            norm2 = float(np.linalg.norm(v3))
            norm3 = float(np.linalg.norm(v5))
            a = [norm1, norm2, norm3]
        if sg == 'hex':
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v3))
            norm3 = float(np.linalg.norm(v2))
            a = [norm1, norm2, norm3]

        for n, i in enumerate(desired_features):
            desired_features[i].append(a[n])

    constants = pd.DataFrame(desired_features)
    for i in desired_features:
        constants.rename(columns={i: fea_class + i}, inplace=True)
    # print(constants.columns.values)

    # print(constants)
    return constants


def density_featurizer(df):
    desired_features = {"density": [], "vpa": [],
                        "packing fraction": []}
    featurizer = DensityFeatures(desired_features)
    fea_class = 'DensityFeatures|'

    for m, s in enumerate(df['structure']):
        a = featurizer.featurize(s)
        for n, i in enumerate(desired_features):
            desired_features[i].append(a[n])

    gsf = pd.DataFrame(desired_features)
    for i in desired_features:
        gsf.rename(columns={i: fea_class + i}, inplace=True)
    # print(gsf.columns.values)

    # print(gsf)
    return gsf


def spacegroup_system_featurizer(df):
    desired_features = {"spacegroup_num": [], "crystal_system": [],
                        "crystal_system_int": [], "is_centrosymmetric": []}
    featurizer = GlobalSymmetryFeatures(desired_features)
    fea_class = 'GlobalSymmetryFeatures|'

    for m, s in enumerate(df['structure']):
        a = featurizer.featurize(s)
        for n, i in enumerate(desired_features):
            desired_features[i].append(a[n])

    gsf = pd.DataFrame(desired_features)
    for i in desired_features:
        gsf.rename(columns={i: fea_class + i}, inplace=True)
    gsf = gsf.drop(columns='GlobalSymmetryFeatures|crystal_system').drop(
        columns='GlobalSymmetryFeatures|is_centrosymmetric')

    # print(gsf)
    return gsf


def remove_constant_feas(df):
        df.drop(df.columns[df.std() == 0], axis=1, inplace=True)
        df = df.loc[:, (df != 0).any(axis=0)]
        return df

def find_Al_MAB_index(all_df):
    all_df = all_df.reset_index()
    idx_list = []
    for idx, structure in zip(all_df['structure'].index, all_df['structure']):
        # print(type(structure))
        eles = [ele.name for ele in structure.composition.elements]
        space = structure.get_space_group_info()
        # print(ele)
        if len(eles) == 3 and ('Al' in eles) and ('B' in eles) and space[1] == 65:
            # print(structure.composition.reduced_formula)
            # print(idx)
            idx_list.append(idx)
    Al_idx_list = [i for i in idx_list if i > len(comps_df)]
    return Al_idx_list


def pick_Al_idx(comps_df):
    Al_idx_list = find_Al_MAB_index(all)
    # print(Al_idx_list)

    comps_df = comps_df.reset_index()
    comp_idx_list = comps_df.index.tolist()
    # print(comp_idx_list)

    train_idx_list = comp_idx_list + Al_idx_list
    # train_idx_list = [int(i) for i in train_idx_list]
    # print(train_idx_list)
    # print(all.index.tolist())
    # train_set = all.loc[idx_list, :]
    # print(len(train_set))

    test_idx_list = [i for i in all.index.tolist() if i not in train_idx_list]
    # print(test_idx_list)
    return train_idx_list, test_idx_list


def get_opt_features_columns(opt_params, opt_dir, name):
    _dict = opt_params.copy()
    for opt in opt_params.keys():
        path = os.path.join(opt_dir, opt + '.csv')
        df = pd.read_csv(path)
        columns = df.columns.values.tolist()
        columns.pop(0)
        columns.pop(-1)
        # print(columns)
        _dict[opt] = columns
    write_json(_dict, os.path.join(opt_dir, name))

def to_csv(path):
        if 'xlsx' in path:
            data = pd.read_excel(path)
        path = path.replace('xlsx', 'csv')
        data = data.set_index('id')
        data.to_csv(path)


if __name__ == '__main__':
    opt_params = {
        'all_5_optimal_features': 5,
        'all_10_optimal_features': 10,
        'all_15_optimal_features': 15,
        'all_20_optimal_features': 20,
        'all_25_optimal_features': 25,
        'all_30_optimal_features': 30,
    }

    # main_label = 'orth'
    # main_label = 'hex'
    main_label = 'hex_orth_space'
    # main_label = 'hex_orth'

    # to_csv(r'G:\codes\modnet\mab_ml\features\comp_space_hex_orth_feas.xlsx')
    # to_csv(r'G:\codes\modnet\mab_ml\features\comp_struc_hex_orth_feas.xlsx')
    # to_csv(r'G:\codes\modnet\mab_ml\features\hex_comp_feas.xlsx')
    # to_csv(r'G:\codes\modnet\mab_ml\features\orth_comp_feas.xlsx')

    # read = False
    read = True
    write = False
    wash = True
    # wash = False

    if main_label:
        if main_label == 'hex_orth':
            mab_df = structures(
                id_json_list=['G:\codes\modnet\mab_ml\data\data\id_CV_outputs_orth.json',
                              'G:\codes\modnet\mab_ml\data\data\id_CV_outputs_hex.json'],
                structures_dir_list=[r'G:\codes\modnet\mab_ml\data\mp_opted_structures_orth',
                                     r'G:\codes\modnet\mab_ml\data\mp_opted_structures_hex']).get_sturctures_set()
            input_name = 'comp_struc_hex_orth_feas.csv'

        if main_label == 'hex_orth_space':
            mab_df = structures(
                id_json_list=['G:\codes\modnet\mab_ml\data\data\id_CV_outputs_orth.json',
                              'G:\codes\modnet\mab_ml\data\data\id_CV_outputs_hex.json'],
                structures_dir_list=[r'G:\codes\modnet\mab_ml\data\mp_opted_structures_orth',
                                     r'G:\codes\modnet\mab_ml\data\mp_opted_structures_hex']).get_sturctures_set()
            input_name = 'comp_space_hex_orth_feas.csv'

        if main_label == 'hex':
            mab_df = structures(id_json_path='G:\codes\modnet\mab_ml\data\data\id_CV_outputs_hex.json',
                                structures_dir=r'G:\codes\modnet\mab_ml\data\mp_opted_structures_hex').get_sturctures_set()
            input_name = 'hex_comp_feas.csv'
        if main_label == 'orth':
            mab_df = structures(id_json_path='G:\codes\modnet\mab_ml\data\data\id_CV_outputs_orth.json',
                                structures_dir=r'G:\codes\modnet\mab_ml\data\mp_opted_structures_orth').get_sturctures_set()
            input_name = 'orth_comp_feas.csv'
        mab_Ed_csv = main_label + '_mab_Ed.csv'
    else:
        'please make features first'

    mab_df = mab_df.reset_index()
    print(mab_df)
    print(mab_df.index)
    print(len(mab_df.index))

    # mab_df = mab_df.iloc[222:248, :]
    # mab_df.to_excel(os.path.join(r'G:\codes\modnet\mab_ml\data', 'mab_data.xlsx'))
    target = mab_df['hull_distance']
    target.index.name = 'id'
    if write == True:
        target.to_csv(os.path.join(r'G:\codes\modnet\mab_ml\data', mab_Ed_csv))

    if read == True:
        if main_label == 'hex_orth':
            feas_name = 'comp_struc_hex_orth_feas'
        if main_label == 'hex_orth_space':
            feas_name = 'comp_space_hex_orth_feas'
        if main_label == 'hex':
            feas_name = 'hex_comp_feas'
        if main_label == 'orth':
            feas_name = 'orth_comp_feas'

        feas_df = pd.read_csv(os.path.join(r'G:\codes\modnet\mab_ml\features', feas_name+'.csv'))
        feas_df = feas_df.set_index(['id'])
        print(feas_df)
    else:
        feas_df = None

    if wash == True:
        feas_df = remove_constant_feas(feas_df)
        print(feas_df)
        print(len(feas_df.columns.values))
        if write == True:
            feas_df.to_csv(os.path.join(r'G:\codes\modnet\mab_ml\features', feas_name+'.csv'))
            df_to_excell(os.path.join(r'G:\codes\modnet\mab_ml\features', feas_name+'.xlsx'), feas_df)

    if write == True:
        from modnet.featurizers.presets import MyFeaturizer

        myfeaturizer = MyFeaturizer()

        get_features(mab_df,
                     feas_df, read_features=read,
                     target_names=['hull_distance'],
                     featurizer=myfeaturizer,
                     # featurizer=None,
                     is_add_lattice=False,
                     is_add_sym=False,
                     # is_opt=False,
                     is_opt=True,
                     opt_feas_limit=50,
                     # feas_name='comp_struc_hex_orth_feas',
                     # feas_name='hex_comp_feas',
                     feas_name='',
                     is_save_feas=False,
                     # is_save_feas=False,
                     for_opt_params=True,
                     # opt_params=None,
                     opt_params=opt_params,
                     opt_feas_name='optimal_features', is_save_opt_feas=True,
                     opt_dir=r'G:\codes\modnet\mab_ml\features\opt_features',
                     fmt='both'
                     )

        ### After feature selection Run this to get columns name in opt subsets
        get_opt_features_columns(opt_params, r'G:\codes\modnet\mab_ml\features\opt_features', 'opt_feas_colums.json')

        ### Then nomalization using normalizer
