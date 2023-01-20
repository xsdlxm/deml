from utils.utils import read_json
import pandas as pd
from utils.utils import read_txt, read_json, write_json
import numpy as np
import scipy.stats


def data_linregress(actual, pred):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(actual, pred)
    rmse = np.sqrt(np.mean(np.square([actual[i] - pred[i] for i in range(len(actual))])))
    mae = np.mean([abs(actual[i] - pred[i]) for i in range(len(actual))])

    return r_value ** 2, rmse, 1-rmse, mae, 1-mae

def read_test_loss(test_log):
    """
    val loss: 0.005299
    """
    test_loss = float(read_txt(test_log)[0][-1])

    # print(test_loss)
    return test_loss


def read_mse_2data(fn):
    """
              epoch  step  train_loss  test_loss
    0       1.0   0.0    2.997552   0.351122
    1       2.0   0.0   14.875697   5.174442
    2       3.0   0.0    5.724105   3.269263
    """
    with open(fn, 'r') as f:
        data = [[m.split(':')[-1].split() for m in i.split('|')] for i in f.readlines()]
        # print(data)
        dd = []
        for m in data:
            aa = []
            for xx in m:
                for ii in xx:
                    aa.append(ii)
            dd.append(aa)

    train_loss = pd.DataFrame(dd, columns=['epoch', 'step', 'train_loss', 'test_loss'])
    train_loss = train_loss.astype(float)
    # return np.array(dd, dtype=np.float64)
    return train_loss


def read_id_tgt_pred_data(test_out):
    test_data = read_txt(test_out)
    # print(test_data)
    test_data = pd.DataFrame(test_data, columns=['id', 'target', 'predict'])
    test_data['target'] = test_data['target'].astype('float64')
    test_data['predict'] = test_data['predict'].astype('float64')
    return test_data


if __name__ == '__main__':
    mab_hull_json = read_json('mab_form_comp_hull_spaceg.json')
    df = pd.DataFrame.from_dict(mab_hull_json).T
    # print(df.columns)
    # ['pretty_formula', 'comps', 'id', 'hull_distance', 'space_group_number']
    df1 = df[df['hull_distance'] < 0.1]
    df1 = df1['hull_distance']
    print(df1)
    print(len(df1))

    # tolearn_df = pd.read_csv(r'G:\codes\modnet\mab_ml\normalizer\clean_feas_target_data_normalized.csv')
    # # print(tolearn_df.columns.values)
    # print(len(tolearn_df.columns.values))
