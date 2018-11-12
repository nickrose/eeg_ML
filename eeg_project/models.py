import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, precision_score,
    recall_score, accuracy_score, f1_score)
from matplotlib import pyplot
import warnings
import tqdm
from eeg_project.read_data import match_types

metric_func_lut = dict(
    acc=accuracy_score,
    auc=roc_auc_score,
    recall=recall_score,
    precision=precision_score,
    f_1_meas=f1_score
)
PRF_metrics = ['recall', 'precision', 'f_1_meas']
basic_metrics = ['acc', 'auc']


def get_variable_learning_rate(nepochs, lr_start_log=-2,
        break_in=[20, 30, 50]):
    """ construct a variable learning rate over the provided number of
        epochs
    """
    assert len(break_in) >= 1

    learning_rate = None

    lr = lr_start_log
    if nepochs > sum(break_in) + 1:
        learning_rate = np.ones(break_in[0]) * np.power(10., lr)

        linear_change_lr = 0
        if len(break_in) > 2:
            linear_change_lr = 1
            learning_rate = np.concatenate([learning_rate,
                np.flip(np.linspace(np.power(10., lr - linear_change_lr),
                    np.power(10., lr), num=break_in[1]), axis=0)], axis=0)
            lr -= linear_change_lr
        if len(break_in) > 1:
            learning_rate = np.concatenate([learning_rate,
                np.ones(break_in[-1]) * np.power(10., lr)], axis=0)

        learning_rate = np.concatenate([learning_rate,
            np.flip(np.logspace(lr - (3 - linear_change_lr), lr,
                                num=(nepochs - sum(break_in))), axis=0)])
    else:
        learning_rate = np.flip(np.logspace(lr-3, lr, num=nepochs), axis=0)

    return np.power(10., lr_start_log), learning_rate


def train_EEG_net(data_tuple, nepochs=10, net=None, metrics2record=None,
        regularization_param=0.2,
        batch_size=32, shuffle_data=True, debug=1):

    if len(data_tuple) > 5:
        multi_class = True
        X_train, y_train, X_test, y_test, uses_cat_feat, \
            classIdDict = data_tuple
    else:
        multi_class = False
        X_train, y_train, X_test, y_test, uses_cat_feat = data_tuple

    dataset = torch.utils.data.TensorDataset(X_train, y_train)

    lr_start, learning_rate = get_variable_learning_rate(nepochs)
    if debug > 2:
        print('learning rate start', lr_start,
            'variable learning rate vector', learning_rate)

    ntrialsXsubjects, _, nsamps, nsen = X_train.shape
    nsamps -= int(uses_cat_feat)

    no_input_model = (net is None)
    if no_input_model:
        net = EEGNet(T=nsamps, C=nsen,
                     noutput_final=(y_train.shape[1] if multi_class else 1),
                     ncat_feat=(len(match_types) if uses_cat_feat else 0))
    if debug > 1:
        trials_test_input = 3
        print('test output', net.forward(Variable(torch.Tensor(
            np.random.rand(trials_test_input, 1, net.T+int(uses_cat_feat),
            net.C)))))

    if multi_class:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    optimizer = optim.Adam(net.parameters(),
        lr=np.power(10., lr_start),
        weight_decay=regularization_param)  # , amsgrad=True)
    if debug > 1:
        print(optimizer)

    if multi_class:
        metrics2record = ["acc"]
    elif metrics2record is None:
        metrics2record = ["acc", "auc", "precision", "recall", "f_1_meas"]

    nmetrics2record = len(metrics2record)

    train_metrics = np.zeros((nepochs, nmetrics2record), dtype=np.float)
    test_metrics = np.zeros((nepochs, nmetrics2record), dtype=np.float)
    loss_metric = np.zeros(nepochs, dtype=np.float)

    if debug == 1:
        progress_bar = tqdm.tqdm(total=nepochs, miniters=1)

    if learning_rate is not None:
        optimizer.param_groups[0]['lr'] = learning_rate[0]

    with warnings.catch_warnings(record=True):
        # Cause all warnings to always be triggered.
        warnings.simplefilter("ignore")

        loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=shuffle_data)
        # order = np.arange(X_train.shape[0])

        # loop over the dataset multiple times
        for epoch in np.arange(nepochs, dtype=int):

            if debug > 1:
                print("\nEpoch ", epoch)

            running_loss = 0.0

            # more basic batching approach
            # if False:
            #     np.random.shuffle(order)
            #     for i in np.arange(len(X_train)//batch_size-1, dtype=int):
            #         s = i*batch_size
            #         e = i*batch_size+batch_size
            #         if isinstance(X_train, torch.Tensor):
            #             train_inputs = X_train[order[s:e]]
            #             train_labels = torch.FloatTensor(np.array(y_train[order[s:e]].numpy()).T * 1.0)
            #         else:
            #             train_inputs = torch.from_numpy(X_train[order[s:e]])
            #             train_labels = torch.FloatTensor(np.array([y_train[order[s:e]]]).T * 1.0)
            #         # wrap them in Variable
            #         train_inputs, train_labels = Variable(train_inputs), Variable(train_labels)

            # use the 'loader' object for batching (shuffling)
            for train_inputs, train_labels in loader:

                # zero the parameter gradients
                optimizer.zero_grad()

                train_inputs = Variable(train_inputs)
                train_labels = Variable(
                    torch.FloatTensor(np.array(train_labels.numpy()) * 1.0))
                # forward + backward + optimize
                outputs = net(train_inputs)

                if multi_class:
                    # train_labels = Variable(torch.LongTensor(
                    #     torch.max(train_labels, 1))[1])
                    # print('train_labels', train_labels.shape)
                    # print(outputs.shape, (torch.max(train_labels, 1)[1]).shape)
                    # print(outputs, Variable((torch.max(train_labels, 1))[1]))
                    loss = criterion(outputs, Variable(torch.LongTensor(
                        torch.max(train_labels, 1)[1])))
                else:
                    loss = criterion(outputs, train_labels)
                # forward + backward + optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

            loss_metric[epoch] = running_loss

            if learning_rate is not None:
                if debug > 2:
                    print('new learning rate', learning_rate[epoch])
                optimizer.param_groups[0]['lr'] = learning_rate[epoch]

            # Check accuracies and other metrics
            train_metrics[epoch, :] = predict_with_model(net, X_train,
                y_train, metrics2record, debug=debug)
            test_metrics[epoch, :] = predict_with_model(net, X_test,
                y_test, metrics2record, debug=debug)

            if debug == 1:
                progress_bar.n = epoch + 1
                progress_bar.set_description(f'loss:{running_loss:.3g}, acc:['
                    f'{train_metrics[epoch, 0]:.2f}|{test_metrics[epoch, 0]:.2f}], '
                    f'lr[{optimizer.param_groups[0]["lr"]:.2g}] ')

            if debug > 1:
                print(metrics2record)
                print("Training Loss ", running_loss)
                print("Train      - ", train_metrics[epoch, :])
        #         print("Validation - ", valid_metrics[epoch, :])
                print("Test       - ", test_metrics[epoch, :])

    # plot loss and other metrics
    # could use: sklearn.model_selection.learning_curve, but this
    # allows less direct control on train/test set split
    if debug:
        pyplot.figure(figsize=(10, 5))
        lg, = pyplot.plot(loss_metric)
        # if learning_rate is not None:
        #     lg, = pyplot.plot(learning_rate)
        pyplot.title('Loss')
        pyplot.xlabel('Epoch')
        pyplot.yscale('log')
        pyplot.show()

        for prm in basic_metrics:
            if prm in metrics2record:
                leg = []
                pyplot.figure(figsize=(10, 5))
                print(prm, metrics2record.index(prm))
                lg, = pyplot.plot(train_metrics[:,
                    metrics2record.index(prm)], label=('train'))
                leg.append(lg)
                lg, = pyplot.plot(test_metrics[:,
                    metrics2record.index(prm)], label=('test'))
                leg.append(lg)

                pyplot.legend(handles=leg)
                pyplot.title(prm)
                pyplot.xlabel('Epoch')
                pyplot.show()

        has_prf = any([(prm in PRF_metrics) for prm in metrics2record])
        if has_prf:
            pyplot.figure(figsize=(10, 5))
            leg = []
            for prm in PRF_metrics:
                if prm in metrics2record:
                    lg, = pyplot.plot(train_metrics[:,
                        metrics2record.index(prm)], label=(prm + ':train'))
                    leg.append(lg)

            for prm in PRF_metrics:
                if prm in metrics2record:
                    lg, = pyplot.plot(test_metrics[:,
                        metrics2record.index(prm)], label=(prm + ':test'))
                    leg.append(lg)

            pyplot.legend(handles=leg)
            pyplot.title('Precision / Recall')
            pyplot.xlabel('Epoch')
            pyplot.show()
    if no_input_model:
        return net, metrics2record, loss_metric, train_metrics, test_metrics
    else:
        return metrics2record, loss_metric, train_metrics, test_metrics


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
        true = np.argmax(Y, axis=1)
    else:
        true = Y
        predicted = np.round(predicted)

    for param in params:
        metric_func = metric_func_lut[param]
        results.append(metric_func(true, predicted))
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
            # print('net shapes', x.shape, cat_feat.shape)
            x = torch.cat([x, cat_feat.reshape(batch_size, 3)], 1)

        x = torch.sigmoid(self.fc1(x))
        # x = F.elu(self.fc1(x))
        # x = self.batchnorm1(x)
        # x = torch.sigmoid(self.fc2(x))
        return x
