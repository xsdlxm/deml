#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import os

def rename_all_dirs_files(new_label, old_label,
                          pths=r'data\pths',
                          opt_nmis=r'features\opt_nmis',
                          opt_features=r'features\opt_features',
                          ml_data_DNNs=r'ml_data\best_DNN',
                          save_dir=r'train\save_dir',
                          test_save_dir=r'test\save_dir',
                          figures=r'plotter\figures',
                          mab_Ed=r'data\mab_Ed.csv',
                          clean_data=r'normalizer\clean_data_normalized.xlsx',
                          clean_data_csv=r'normalizer\clean_data_normalized.csv',
                          ml_out=r'data\data\ml_output.json', ):
    dirs = [pths, opt_nmis, opt_features, ml_data_DNNs, save_dir, test_save_dir, figures, clean_data, clean_data_csv,
            ml_out, mab_Ed]
    # dirs = [r'G:\codes\modnet\zwm']
    from pathlib import Path
    for d in dirs:
        path = Path(d)
        parent = path.parent
        name = path.name
        print(parent)
        print(name)
        if old_label:
            name = old_label + '_' + name
            path = Path(os.path.join(parent, name))
        new_name = new_label + '_' + name
        new_path = os.path.join(parent, new_name)

        if not os.path.exists(new_path):
            path.rename(new_path)
        else:
            pass


if __name__ == '__main__':
    # rename_all_dirs_files(r'hex_orth')
    # rename_all_dirs_files(r'hex')
    # rename_all_dirs_files(r'orth')
    rename_all_dirs_files(r'hex_orth_space', None)
    # rename_all_dirs_files(r'test_old', r'selected1')