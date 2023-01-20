import pandas as pd
from utils. utils import df_to_excell, read_json

if __name__=='__main__':
    raw_features = pd.read_csv('new_structure_composition_features.csv')
    fea_names = raw_features.columns.values
    print(len(fea_names))
    # opt_cross_nmi = pd.read_csv('opt_nmis\optimal_features_cross_nmi.csv')
    # opt_cross_nmi = opt_cross_nmi.set_index(opt_cross_nmi.columns.values[0])
    # # print(opt_cross_nmi)
    #
    # opt_target_nmi = pd.read_csv('opt_nmis\optimal_features_target_nmi_.csv')
    # # print(opt_target_nmi)
    # opt_target_nmi = opt_target_nmi.sort_values(by="hull_distance", ascending=False).set_index(opt_target_nmi.columns[0])
    # opt_target_nmi.index.name = 'features'
    # # print(opt_target_nmi)
    # # df_to_excell('opt_nmis\sorted_tgt_nmi.xlsx', opt_target_nmi)
    #
    # opt_feas_json = read_json(r'G:\codes\modnet\mab_ml\features\opt_features\opt_feas_colums.json')
    # # opt_feas_name = list(opt_feas_json.values())[0]
    # opt_feas_name = opt_feas_json['all_60_optimal_features']
    # print(opt_feas_name)
    #
    # # for name in opt_feas_name:
    # #     print(name)
    # #     tgt_nmi = opt_target_nmi.loc[name, opt_target_nmi.columns.values[-1]]
    # #     print(tgt_nmi)
    #
    # n = 1
    # fea1 = opt_feas_name[n-1]
    # group_size = 20
    # cross_compare = opt_feas_name[n-1: n-1+group_size]
    # for fea in cross_compare:
    #     print(fea)
    #     cross_nmi = opt_cross_nmi.loc[fea1, fea]
    #     print(cross_nmi)




