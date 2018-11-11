import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, precision_score,
    recall_score, accuracy_score, f1_score)

metric_func_lut = dict(
    acc=accuracy_score,
    auc=roc_auc_score,
    recall=recall_score,
    precision=precision_score,
    f_1_meas=f1_score
)
PRF_metrics = ['recall', 'precision', 'f_1_meas']
basic_metrics = ['acc', 'auc']


def predict_with_model(model, X, Y, params=["acc"], debug=1):
    results = []
    nsamples = X.shape[0]
    batch_size = int(2 * np.sqrt(nsamples))
    while (nsamples//batch_size) * batch_size != nsamples:
        batch_size += 1
    if debug > 1:
        print(f'selected batch size: {batch_size}')

    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    else:
        assert not(isinstance(X, torch.Tensor)), (
            "cannot have mixed torch.Tensor, and numpy types")
        X = torch.from_numpy(X)

    inputs = Variable(X)
    predicted = model(inputs)
    predicted = predicted.data.cpu().numpy()

    if predicted.shape[1] > 1:  # if multi-class, get the index of the correct label
        predicted = np.argmax(predicted, axis=1)
    for param in params:
        metric_func = metric_func_lut[param]
        results.append(metric_func(np.argmax(Y, axis=1), predicted))
    return results


def predict_twoclass_with_model(model, X, Y, params=["acc"], debug=1):
    """ predict the model on the given data and use the labels to
        return the metrics of choice
    """
    results = []
    nsamples = X.shape[0]
    batch_size = int(2 * np.sqrt(nsamples))
    while (nsamples//batch_size) * batch_size != nsamples:
        batch_size += 1
    if debug > 1:
        print(f'selected batch size: {batch_size}')

    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    else:
        assert not(isinstance(X, torch.Tensor)), (
            "cannot have mixed torch.Tensor, and numpy types")
        X = torch.from_numpy(X)

    predicted = []
    if nsamples > batch_size:
        for i in range(nsamples//batch_size):
            s = i*batch_size
            e = i*batch_size+batch_size

            if isinstance(X, torch.Tensor):
                inputs = Variable(X[s:e])
            else:
                inputs = Variable(torch.from_numpy(X[s:e]))

            pred = model(inputs)
            predicted.extend(pred.data.cpu().numpy().tolist())

        predicted = np.asarray(predicted)[:, 0]

    else:
        inputs = Variable(X)
        predicted = model(inputs)
        predicted = predicted.data.cpu().numpy()

    for param in params:
        metric_func = metric_func_lut[param]
        results.append(metric_func(Y, np.round(predicted)))
    return results


class EEGNet(nn.Module):
    """ Implement a convolutional NN according to:

        Lawhern et al., EEGNet: A Compact Convolutional Network for
        EEG-based Brain-Computer Interfaces, 2018
        https://arxiv.org/abs/1611.08024

    """
    def __init__(self, C=64, T=256, ncat_feat=0, noutput_final=1):
        super(EEGNet, self).__init__()
        self.T = T
        self.C = C
        self.ncat_feat = ncat_feat

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, C), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        # NOTE: use the ZeroPad2d functional since we want
        # non-uniform padding, otherwise padding can be specified
        # in the Conv2d functional
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, C//2))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of samples
        interconnected_fc_nodes = T//2 + ncat_feat
        self.fc1 = nn.Linear(interconnected_fc_nodes,
            # 2 + interconnected_fc_nodes)
            noutput_final)
        # self.fc2 = nn.Linear(2 + interconnected_fc_nodes, noutput_final)

    def forward(self, x):
        # subselect_real_features
        if self.ncat_feat:
            x, cat_feat = (x[:, :, :self.T, :self.C],
                x[:, :, self.T, 0:self.ncat_feat])
            batch_size = x.shape[0]

        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, self.T//2)
        if self.ncat_feat:
            print('net shapes', x.shape, cat_feat.shape)
            x = torch.cat([x, cat_feat.reshape(batch_size, 3)], 1)

        x = torch.sigmoid(self.fc1(x))
        # x = F.elu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        return x
