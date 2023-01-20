#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import os, time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_json, write_json, df_to_excell, check_dir_existence
from torch.utils.data.sampler import RandomSampler
from data.train_label_info import train_for_params


def opt_columns(opt_features_path):
    with open(opt_features_path, 'r') as f:
        con = f.readline().split(',')
        con.pop(0)
        con.pop(-1)
        # print(con)
    return con


def get_opt_columns(opt_features_dir, fmt):
    ls = os.listdir(opt_features_dir)
    print(ls)
    colums = {}
    for l in ls:
        try:
            if fmt == 'csv':
                data = pd.read_csv(os.path.join(opt_features_dir, l))
            if fmt == 'xlsx':
                data = pd.read_excel(os.path.join(opt_features_dir, l))
            data = data.drop(columns='Unnamed: 0')
            data_columns = data.columns.values
            print(len(data_columns))
            colums[l.split('.')[0]] = data_columns
        except:
            print('not ok?')
            print(l)


class MyDataset(Dataset):
    def __init__(self, file_name, opt_num, feas_colums_json):
        feas_colums = read_json(feas_colums_json)
        print('opt_num: ', opt_num)
        opt_feas_file_name = 'all_' + str(opt_num) + '_optimal_features'
        colums = feas_colums[opt_feas_file_name]

        # x = price_df.iloc[:, 1:col_n - 1].values
        # opt_csv = os.path.join(opt_dir, )
        # cols = opt_columns(opt_csv)

        if '.csv' in file_name:
            price_df = pd.read_csv(file_name, header=0)
        if '.xlsx' in file_name:
            price_df = pd.read_excel(file_name, header=0)

        if 'id' in price_df.columns.values:
            self.id = price_df['id'].values
            price_df = price_df.drop('id', axis=1)
        else:
            id = price_df.index.values
            self.id = id
        print(price_df)
        col_n = len(price_df.columns)
        # print(self.id)

        x = price_df.loc[:, colums].values
        print(len(x))
        y = price_df.iloc[:, col_n - 1].values
        # print(id)
        # print(type(x))
        print(y)

        self.x_train = torch.tensor(x, dtype=torch.float64)
        # self.x_train = torch.tensor(x)
        self.y_train = torch.tensor(y, dtype=torch.float64)
        # self.y_train = torch.tensor(y)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.id[idx]


class DNN(nn.Module):
    DP_RADIO = 0.5
    B_ININT = -0.0

    def __init__(self, n_feature, n_hidden, n_output, batch_normalize=True, dropout=True, activation=nn.ReLU()):
        super(DNN, self).__init__()
        assert isinstance(n_hidden, (list, tuple))

        self.ACTIVATION = activation
        self.do_bn = batch_normalize
        self.do_dp = dropout
        self.fcs, self.bns, self.dps = [], [], []

        self.bn_input = nn.BatchNorm1d(n_feature, momentum=0.5)
        self.n_hidden = [n_feature] + n_hidden

        for hid_index in range(1, len(self.n_hidden)):
            fc = torch.nn.Linear(self.n_hidden[hid_index - 1], self.n_hidden[hid_index])
            setattr(self, 'fc%d' % hid_index, fc)
            self._set_init(fc)
            self.fcs.append(fc)

            if self.do_bn:
                bn = nn.BatchNorm1d(self.n_hidden[hid_index], momentum=0.5)
                setattr(self, 'bn%d' % hid_index, bn)
                self.bns.append(bn)

            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'dp%s' % hid_index, dp)
                self.dps.append(dp)

        self.predict = torch.nn.Linear(self.n_hidden[-1], n_output)
        self._set_init(self.predict)

    def _set_init(self, fc):
        nn.init.normal_(fc.weight, mean=0, std=0.1)
        nn.init.constant_(fc.bias, self.B_ININT)

    def forward(self, x):
        # if self.do_bn: x = self.bn_input(x)
        for i in range(len(self.n_hidden) - 1):
            x = self.fcs[i](x)
            if self.do_bn:
                x = self.bns[i](x)
            if self.do_dp:
                x = self.dps[i](x)
            x = self.ACTIVATION(x)

        x = self.predict(x)
        return x


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def do_time():
#     return now_time().replace(' ', '_').replace('-', '_').replace(':', '_')


def train(file_path='',
          cuda=False,
          epoch=1000,
          save_dir='',
          label='',
          load_data=False,
          hidden_nodes=None,
          activation=None,
          optimizer=None,
          opt_feas_num=None,
          save_module=True,
          save_step=100,
          restore=False,
          module_params_fn=None,
          lr=0.01,
          batch_normalize=True,
          dropout=True,
          train_size=0.8,
          val_size=0.1,
          batch_size=2000,
          shuffle=True,
          workers=0,
          loss_limit=1000

          ):
    rember_loss = 0
    _go = True

    check_dir_existence(save_dir)
    pths_dir = r'data\pths'
    check_dir_existence(pths_dir)

    new_label = opt_feas_num + '_' + label
    new_opt_num_dir = os.path.join(save_dir, new_label)
    check_dir_existence(new_opt_num_dir)

    if hidden_nodes is None:
        hidden_nodes = [100, 50, 50, 20]

    # sampler = RandomSampler(dataset)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler, pin_memory=True)
    #
    # for x in loader:
    #     print(x)
    #     break
    #
    # sampler2 = RandomSampler(dataset)
    # torch.save(sampler.generator, "test_samp.pth")
    # sampler2.set_state(torch.load("test_samp.pth"))
    # loader2 = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler2, pin_memory=True)
    #
    # for x in loader2:
    #     print(x)

    if load_data == True:
        train_loader = torch.load('data/pths/train_loader.pth')
        val_loader = torch.load('data/pths/val_loader.pth')
        test_loader = torch.load('data/pths/test_loader.pth')
    else:
        # load data
        dataset = MyDataset(file_path, opt_feas_num,
                            'features\opt_features\opt_feas_colums.json')

        train_size = int(len(dataset) * train_size)
        validate_size = int(len(dataset) * val_size)
        test_size = len(dataset) - validate_size - train_size
        # test_size = len(dataset) - train_size

        train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                      # train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                      [train_size, validate_size,
                                                                                       test_size])
        #                                                             [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        val_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

        torch.save(train_loader, 'data/pths/train_loader_%s.pth' % new_label)
        torch.save(val_loader, 'data/pths/val_loader_%s.pth' % new_label)
        torch.save(test_loader, 'data/pths/test_loader_%s.pth' % new_label)

    # print(train_loader.dataset[0][1])
    n_feature = train_loader.dataset[0][0].shape[0]
    # print(len(train_loader.dataset[0][0]))
    print('n_features: ', n_feature)

    if cuda:
        dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=1,
                  batch_normalize=batch_normalize, dropout=dropout, activation=activation).cuda()
    else:
        dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=1,
                  batch_normalize=batch_normalize, dropout=dropout, activation=activation)

    if restore:
        dnn.load_state_dict(torch.load(module_params_fn))
    print(dnn)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(dnn.parameters(), lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(dnn.parameters(), lr)
    else:
        raise ValueError("Only support Adam and SGD")

    loss_func = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in dnn.parameters():
    #     reg_loss += l1_crit(param, 100)
    tfn = os.path.join(new_opt_num_dir, 'running_%s.log' % new_label)

    for ep in range(epoch):
        epoch = ep + 1
        # print("epoch: %s" % epoch)
        # if epoch % 1500 == 0:
        #     lr = lr * 0.5
        #     adjust_learning_rate(optimizer, lr)

        for step, (b_x, b_y, id) in enumerate(train_loader):
            # print("train step: %s" % step)
            # input_data = torch.DoubleTensor(b_x)
            # print(input_data)
            # print(input_data.shape)
            # print(dnn.fc1.weight.grad)
            if cuda:
                b_x, b_y = b_x.cuda(), b_y.cuda()
            else:
                b_x, b_y = b_x, b_y
            output = dnn(b_x.float())
            id = id.reshape(-1, 1)
            label_y = b_y.reshape(-1, 1)
            loss = loss_func(output, label_y.float())  # + 0.005 * reg_loss

            with open(os.path.join(new_opt_num_dir, new_label + '_train.out'), 'w') as f:
                for i in range(len(label_y.data.numpy())):
                    f.write("%d    %.7f    %.7f\n" % (
                        int(id.data.numpy()[i][0]), label_y.data.numpy()[i][0], output.data.numpy()[i][0]))
                    # print(id.data.numpy()[i][0], '   ', label_y.data.numpy()[i][0], '   ', output.data.numpy()[i][0])

            dnn.eval()
            for _, (val_x, val_y, val_id) in enumerate(val_loader):

                if cuda:
                    val_x, val_y = val_x.cuda(), val_y.cuda()

                val_output = dnn(val_x.float())
                val_id = val_id.reshape(-1, 1)
                val_label_y = val_y.reshape(-1, 1)
                val_loss = loss_func(val_output, val_label_y.float())  # + 0.005 * reg_loss

                # print('val_loss: ', val_loss.data.numpy())
                with open(os.path.join(new_opt_num_dir, new_label + '_validation.out'), 'w') as f:
                    for i in range(len(val_label_y.data.numpy())):
                        f.write("%d    %.7f    %.7f\n" % (
                            int(val_id.data.numpy()[i][0]), val_label_y.data.numpy()[i][0],
                            val_output.data.numpy()[i][0]))
                        # print(val_id.data.numpy()[i][0], '   ', val_label_y.data.numpy()[i][0], '   ',
                        #       val_output.data.numpy()[i][0])

            # print(output.cpu().data.numpy().shape, val_label_y.cpu().data.numpy().shape)
            if rember_loss != 0:
                # if ((val_loss.cpu().data.numpy() - rember_loss) > 0.1) or ((val_loss.cpu().data.numpy() - loss.cpu().data.numpy()) > 0.02):
                if (val_loss.cpu().data.numpy() - rember_loss) > loss_limit:
                    _go = False
            else:
                _go = True

            txt_temple = 'Epoch: {0} | Step: {1} | ' \
                         'train loss: {2:.6f} | ' \
                         'val loss: {3:.6f}'.format(epoch, step,
                                                    loss.cpu().data.numpy(),
                                                    val_loss.cpu().data.numpy())
            # print(txt_temple)
            # now_step = step + epoch * math.ceil(TOTAL_LINE / BATCH_SIZE)
            if epoch == 0:
                write(tfn, txt_temple, 'w')
            else:
                # if now_step % SAVE_STEP == 0:
                write(tfn, txt_temple, 'a')

            if save_module:
                if epoch % save_step == 0:
                    torch.save(dnn, os.path.join(new_opt_num_dir, 'dnn_%d_%s.pkl' % (epoch, new_label)))
                    torch.save(dnn.state_dict(),
                               os.path.join(new_opt_num_dir, 'dnn_params_%d_%s.pkl' % (epoch, new_label)))

            if _go:
                rember_loss = val_loss.cpu().data.numpy()
                dnn.train()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                return None

    time.sleep(1)
    ### run_test
    ttest(test_loader, mp_fn=os.path.join(new_opt_num_dir, 'dnn_params_%d_%s.pkl' % (epoch, new_label)),
          test_dir=r'test\save_dir',
          output_fn='results_%d_%s_test.out' % (epoch, new_label),
          loss_log='%s_test_loss.log' % new_label,
          hidden_nodes=hidden_nodes, activation=activation, n_output=1,
          batch_normalize=True, dropout=True)


def ttest(test_loader=None, mp_fn='', test_dir='', output_fn='', loss_log='',
          hidden_nodes=None, activation=nn.ReLU(), n_output=1,
          batch_normalize=True,
          dropout=True,
          ):
    if os.path.exists(test_dir) == False:
        if __name__ == '__main__':
            os.mkdir(test_dir)

    if test_loader == None:
        test_loader = torch.load('data/test_loader.pth')
    else:
        test_loader = test_loader

    if hidden_nodes is None:
        hidden_nodes = [100, 50, 50, 20]

    n_feature = test_loader.dataset[0][0].shape[0]
    print(n_feature)

    dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=n_output,
              batch_normalize=batch_normalize, dropout=dropout, activation=activation)

    dnn.load_state_dict(torch.load(mp_fn))
    loss_func = nn.MSELoss()

    for step, (b_x, b_y, id) in enumerate(test_loader):
        b_x, b_y = b_x, b_y
        dnn.eval()
        output = dnn(b_x.float())
        id = id.reshape(-1, 1)
        label_y = b_y.reshape(-1, 1)
        loss = loss_func(output, label_y.float())
        # print('loss: ', loss.data.numpy(), 'label_y: ', label_y.data.numpy(), 'predict_y: ', output.data.numpy())
        print('test_loss: ', loss.data.numpy())

        # from utils.utils import write_json
        # results = {}
        for i in range(len(label_y.data.numpy())):
            #     # print(id.data.numpy()[i][0])
            #     # print(label_y.data.numpy()[i][0])
            #     # print(output.data.numpy()[i][0])
            #     results[id.data.numpy()[i][0]] = {'target': label_y.data.numpy()[i][0],
            #                                       'output': output.data.numpy()[i][0]}
            # print(output_fn)
            with open(os.path.join(test_dir, output_fn), 'w') as f:
                for i in range(len(label_y.data.numpy())):
                    f.write("%d    %.7f    %.7f\n" % (
                        int(id.data.numpy()[i][0]), label_y.data.numpy()[i][0], output.data.numpy()[i][0]))
                    # print(id.data.numpy()[i][0], '    ', label_y.data.numpy()[i][0], '    ', output.data.numpy()[i][0])

        tfn = os.path.join(test_dir, loss_log)
        txt_temple = 'val loss: {0:.6f}'.format(loss.cpu().data.numpy())
        write(tfn, txt_temple, 'w')
        # if step == 10:
        #     break

    #     results = {'values': results, 'loss': loss}
    #     print(results)
    #
    # write_json(results, os.path.join(test_dir, output_fn+'.json'))


def cross_entropyloss_ntype_ttest(file_path, mp_fn, save_dir='', output_fn='', n_feature=34, shuffle=False,
                                  hidden_nodes=None, activation=nn.ReLU(), batch_size=252, zt=False, workers=0,
                                  n_output=1, has_t=None):
    # csv_fn = r'G:\ztml\ztml\data\clean_data_normalized.csv'
    # test_csv_fn = r'G:\ztml\ztml\data\test_data_from_normalized_data.csv'
    if hidden_nodes is None:
        hidden_nodes = [100, 50, 50, 20]
    # load data
    test_dataset = MyDataset(file_path, 200)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

    dnn = DNN(n_feature=n_feature, n_hidden=hidden_nodes, n_output=n_output, batch_normalize=True, dropout=True,
              activation=activation)

    dnn.load_state_dict(torch.load(mp_fn))
    loss_func = nn.CrossEntropyLoss()

    for step, (b_x, b_y) in enumerate(test_loader):
        b_x, b_y = b_x, b_y
        dnn.eval()
        output = dnn(b_x.float())

        label_y = b_y.reshape(-1, 1)
        # label_y = b_y.long()
        loss = loss_func(output, b_y.long())
        # print('loss: ', loss.data.numpy(), 'label_y: ', label_y.data.numpy(), 'predict_y: ', output.data.numpy())
        print('loss: ', loss.data.numpy())

        def change_output(x):
            if x[0] > x[1]:
                return 0.0
            else:
                return 1.0

        with open(os.path.join(save_dir, output_fn), 'w') as f:
            for i in range(len(label_y.data.numpy())):
                if has_t:
                    if has_t is not None:
                        f.write("%.7f     %.7f      %s\n" % (
                            label_y.data.numpy()[i][0], change_output(output.data.numpy()[i]),
                            '  '.join([str(b_x.data.numpy()[i][m]) for m in has_t])))
                    else:
                        f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], change_output(output.data.numpy()[i])))
                else:
                    f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], change_output(output.data.numpy()[i])))
                # print(label_y.data.numpy()[i][0], '   ', change_output(output.data.numpy()[i]))


def write(fn, content, mode='w'):
    with open(fn, mode) as f:
        f.write(content + '\n')


sele = lambda x: x


def run_train_test(
        file_path=None,
        save_dir=None,
        loss_limit=100,
        load_data=False,
        labels=None, activations=None, hidden_layers=None, optimizers=None, opt_feas_nums=None
):
    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        opt_feas_num = opt_feas_nums[i]

        # if i >= 3:
        #     epoch = 12000
        #     save_step = 100
        # else:
        #     epoch = 1000
        #     save_step = 100

        epoch = 1000
        save_step = 100

        train(
            file_path=file_path,
            cuda=False,
            epoch=epoch,
            save_dir=save_dir,
            label=label,
            load_data=load_data,
            hidden_nodes=hidden_layer,
            activation=activation,
            optimizer=optimizer,
            opt_feas_num=opt_feas_num,
            save_module=True,
            save_step=save_step,
            restore=False,
            module_params_fn=None,
            lr=0.01,
            batch_normalize=True,
            dropout=True,
            train_size=0.6,
            val_size=0.1,
            batch_size=2000,
            shuffle=True,
            workers=0,
            loss_limit=loss_limit
        )


def run_test(file_path=None,
             save_dir=None,
             ):
    labels = ["600_400_layer_1000_relu"]
    activations = [nn.ReLU()]
    hidden_layers = [[600, 400]]
    opt_feas_nums = ['200']

    # labels = ["600_400_layer_1000_relu", "600_400_layer_1000_sigmoid", "600_400_layer_1000_tanh",
    #           "600_400_layer_1000_relu_sgd"]
    # activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU()]
    # hidden_layers = [[600, 400], [600, 400], [600, 400], [600, 400]]
    # optimizers = ['Adam', 'Adam', 'Adam', 'SGD']
    # opt_feas_nums = ['200', '200', '200', '200']

    labels = sele(labels)
    activations = sele(activations)
    hidden_layers = sele(hidden_layers)
    opt_feas_nums = sele(opt_feas_nums)

    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        opt_feas_num = opt_feas_nums[i]

        if i >= 3:
            epoch = 12000
        else:
            epoch = 1000

        new_label = opt_feas_num + '_' + label
        new_opt_num_dir = os.path.join(save_dir, new_label)
        if os.path.isdir(new_opt_num_dir):
            pass
        else:
            os.mkdir(new_opt_num_dir)

        ttest(test_loader=file_path, mp_fn=os.path.join(new_opt_num_dir, 'dnn_params_%d_%s.pkl' % (epoch, new_label)),
              test_dir=new_opt_num_dir,
              output_fn='results_%s_test.out' % new_label,
              loss_log='%s_test_loss.log' % new_label,
              hidden_nodes=hidden_layer, activation=activation, n_output=1,
              batch_normalize=True, dropout=True)


def rename_all_dirs_files(new_label,
                          pths=r'data\pths',
                          opt_nmis=r'features\opt_nmis',
                          opt_features=r'features\opt_features',
                          ml_data_DNNs=r'ml_data\Ed\MAB\DNN',
                          save_dir=r'train\save_dir',
                          test_save_dir=r'test\save_dir',
                          figures=r'plotter\figures',

                          clean_data=r'normalizer\clean_all_feas_target_data_normalized.xlsx',
                          clean_data_csv=r'normalizer\clean_feas_target_data_normalized.csv',
                          ml_out=r'data\data\ml_output.json', ):
    dirs = [pths, opt_nmis, opt_features, ml_data_DNNs, save_dir, test_save_dir, figures, clean_data, clean_data_csv,
            ml_out]
    # dirs = [r'G:\codes\modnet\zwm']
    from pathlib import Path
    for d in dirs:
        path = Path(d)
        parent = path.parent
        name = path.name
        print(parent)
        print(name)
        new_name = new_label + '_' + name
        new_path = os.path.join(parent, new_name)

        if not os.path.exists(new_path):
            path.rename(new_path)
        else:
            pass


if __name__ == '__main__':
    # rename_all_dirs_files(r'dir label')

    file_path = r'normalizer\clean_data_normalized.csv'
    save_dir = r'train\save_dir'
    test_save_dir = r'test\save_dir'
    check_dir_existence(save_dir)
    check_dir_existence(test_save_dir)

    # nums = [i for i in range(20, 220, 20)] + [30, 50, 250] + [i for i in range(300, 1000, 100)]
    opt_params = {
        'all_5_optimal_features': 5,
        'all_10_optimal_features': 10,
        'all_15_optimal_features': 15,
        'all_20_optimal_features': 20,
        'all_25_optimal_features': 25,
        'all_30_optimal_features': 30,
    }

    hls = [[100, 50], [200, 100], [400, 200], [600, 400], [100, 50, 20], [200, 100, 40], [400, 200, 80],
           [100, 50, 20, 10], [200, 100, 40, 20]]
    # hls = [[100, 50], [200, 100]]
    nums = list(opt_params.values())
    for hl in hls:
        labels, activations, hidden_layers, optimizers, opt_feas_nums = train_for_params(nums, hl)
        run_train_test(file_path, save_dir, 2000, False, labels, activations, hidden_layers, optimizers, opt_feas_nums)
