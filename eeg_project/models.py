import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class EEGNet(nn.Module):
    """ implement a convolution NN according the idea laid out in
        Lawhern et al., EEGNet: A Compact Convolutional Network for EEG-based
            Brain-Computer Interfaces, 2018
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
        # NOTE: This dimension will depend on the number of samples in time: T
        self.fc1 = nn.Linear(T//2 + ncat_feat, noutput_final)

    def forward(self, x):
        # subselect_real_features
        if self.ncat_feat:
            x, cat_feat = (x[:, :, :self.T, :self.C],
                x[:, :, self.T, self.C:(self.C + self.ncat_feat)])

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
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
            x = torch.cat([x, cat_feat],)
        x = torch.sigmoid(self.fc1(x))
#         print(x.__class__)
        return x
