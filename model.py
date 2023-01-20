#   coding:utf-8

__author__ = 'xsdlxm'
__version__ = 1.0
__maintainer__ = 'xsdlxm'
__email__ = "1041847987@qq.com"
__date__ = '2022/03/08 15:36:27'

import torch
import torch.nn as nn

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

if __name__ == '__main__':
    dnn = DNN(n_feature=30, n_hidden=[100, 50, 50, 20], n_output=1,
              batch_normalize=True, dropout=True, activation=nn.ReLU())
    print(dnn)