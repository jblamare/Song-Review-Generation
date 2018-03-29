import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from settings import number_labels


class local_model(nn.Module):
    def __init__(self, segment_size):
        super(local_model, self).__init__()
        self.segment_size = segment_size
        # convs and norms
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_features=128)
        if self.segment_size == 18:
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
            self.norm3 = nn.BatchNorm1d(num_features=256)
        if self.segment_size >= 27:
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.norm3 = nn.BatchNorm1d(num_features=256)
            if self.segment_size >= 54:
                self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, padding=1)
                self.norm4 = nn.BatchNorm1d(num_features=256)
                if self.segment_size >= 108:
                    self.conv5 = nn.Conv1d(in_channels=256, out_channels=256,
                                           kernel_size=2, padding=1)
                    self.norm5 = nn.BatchNorm1d(num_features=256)
                    if self.segment_size == 216:
                        self.conv6 = nn.Conv1d(
                            in_channels=256, out_channels=256, kernel_size=2, padding=1)
                        self.norm6 = nn.BatchNorm1d(num_features=256)

        # pools
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        if self.segment_size == 18:
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.pool6 = nn.MaxPool1d(kernel_size=6)
            self.pool18 = nn.MaxPool1d(kernel_size=18)
        if self.segment_size == 27:
            self.pool9 = nn.MaxPool1d(kernel_size=9)
            self.pool27 = nn.MaxPool1d(kernel_size=27)
        if self.segment_size == 54:
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.pool6 = nn.MaxPool1d(kernel_size=6)
            self.pool18 = nn.MaxPool1d(kernel_size=18)
            self.pool54 = nn.MaxPool1d(kernel_size=54)
        if self.segment_size == 108:
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.pool4 = nn.MaxPool1d(kernel_size=4)
            self.pool12 = nn.MaxPool1d(kernel_size=12)
            self.pool36 = nn.MaxPool1d(kernel_size=36)
            self.pool108 = nn.MaxPool1d(kernel_size=108)
        if self.segment_size == 216:
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.pool4 = nn.MaxPool1d(kernel_size=4)
            self.pool8 = nn.MaxPool1d(kernel_size=8)
            self.pool24 = nn.MaxPool1d(kernel_size=24)
            self.pool72 = nn.MaxPool1d(kernel_size=72)
            self.pool216 = nn.MaxPool1d(kernel_size=216)
        self.avgpool = nn.AvgPool1d(kernel_size=1255//self.segment_size)
        # linears
        self.linearOut1 = nn.Linear(in_features=256, out_features=256)
        self.dropout = nn.Dropout(0.5)
        self.linearOut2 = nn.Linear(in_features=256, out_features=number_labels)

    def forward(self, X):
        X1 = self.conv1(X)  # BS*128*1242 (18*69 // 27*46 // 54*23) // 1188 (108*11) // 1080 (216*5)
        X = self.pool3(F.relu(self.norm1(X1)))
        X2 = self.conv2(X)  # BS*128*414 (6*69 // 9*46 // 18*23) // 396 (36*11) // 360 (72*5)
        X = self.pool3(F.relu(self.norm2(X2)))
        if self.segment_size == 18:
            X3 = self.conv3(X)  # BS*256*138 (2*69)
            X = self.pool2(F.relu(self.norm3(X3)))
        if self.segment_size >= 27:
            X3 = self.conv3(X)  # BS*256*138 (3*46 // 6*23) // 132 (12*11) // 120 (24*5)
            X = self.pool3(F.relu(self.norm3(X3)))
            if self.segment_size >= 54:
                X4 = self.conv4(X)  # BS*256*46 (2*23) // 44 (4*11) // 40 (8*5)
                X = self.pool2(F.relu(self.norm4(X4)))
                if self.segment_size >= 108:
                    X5 = self.conv5(X)  # BS*256*22 (2*11) // 20 (4*5)
                    X = self.pool2(F.relu(self.norm5(X5)))
                    if self.segment_size == 216:
                        X6 = self.conv6(X)  # BS*256*10 (2*5)
                        X = self.pool2(F.relu(self.norm6(X6)))
        X = self.avgpool(X).squeeze(2)
        X = F.relu(self.linearOut1(X))
        X = self.dropout(X)
        X = self.linearOut2(X)

        if self.segment_size == 18:
            X1 = self.avgpool(self.pool18(X1)).squeeze(2)  # 64*128
            X2 = self.avgpool(self.pool6(X2)).squeeze(2)  # 64*128
            X3 = self.avgpool(self.pool2(X3)).squeeze(2)  # 64*256
        if self.segment_size == 27:
            X1 = self.avgpool(self.pool27(X1)).squeeze(2)  # 64*128
            X2 = self.avgpool(self.pool9(X2)).squeeze(2)  # 64*128
            X3 = self.avgpool(self.pool3(X3)).squeeze(2)  # 64*256
        if self.segment_size == 54:
            X1 = self.avgpool(self.pool54(X1)).squeeze(2)  # 64*128
            X2 = self.avgpool(self.pool18(X2)).squeeze(2)  # 64*128
            X3 = self.avgpool(self.pool6(X3)).squeeze(2)  # 64*256
            X4 = self.avgpool(self.pool2(X4)).squeeze(2)  # 64*256
        if self.segment_size == 108:
            X1 = self.avgpool(self.pool108(X1)).squeeze(2)  # 64*128
            X2 = self.avgpool(self.pool36(X2)).squeeze(2)  # 64*128
            X3 = self.avgpool(self.pool12(X3)).squeeze(2)  # 64*256
            X4 = self.avgpool(self.pool4(X4)).squeeze(2)  # 64*256
            X5 = self.avgpool(self.pool2(X5)).squeeze(2)  # 64*256
        if self.segment_size == 216:
            X1 = self.avgpool(self.pool216(X1)).squeeze(2)  # 64*128
            X2 = self.avgpool(self.pool72(X2)).squeeze(2)  # 64*128
            X3 = self.avgpool(self.pool24(X3)).squeeze(2)  # 64*256
            X4 = self.avgpool(self.pool8(X4)).squeeze(2)  # 64*256
            X5 = self.avgpool(self.pool4(X5)).squeeze(2)  # 64*256
            X6 = self.avgpool(self.pool2(X6)).squeeze(2)  # 64*256

        if self.segment_size == 18:
            X_cat = torch.cat((X1, X2, X3), dim=1)  # 64*512
        if self.segment_size == 27:
            X_cat = torch.cat((X1, X2, X3), dim=1)  # 64*512
        if self.segment_size == 54:
            X_cat = torch.cat((X1, X2, X3, X4), dim=1)  # 64*768
        if self.segment_size == 108:
            X_cat = torch.cat((X1, X2, X3, X4, X5), dim=1)  # 64*1024
        if self.segment_size == 216:
            X_cat = torch.cat((X1, X2, X3, X4, X5, X6), dim=1)  # 64*1280
        return X, X_cat


class global_model(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(global_model, self).__init__()
        self.lin1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.lin2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.lin3 = nn.Linear(in_features=n_hidden, out_features=number_labels)

    def forward(self, X):
        X1 = F.relu(self.lin1(X))
        X2 = F.relu(self.lin2(X1))
        X = self.lin3(X2)
        return X, X2
