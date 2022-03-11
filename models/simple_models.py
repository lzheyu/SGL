import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDimConvolution(nn.Module):
    def __init__(self, subgraph_num, hop_num, feat_dim):
        super(OneDimConvolution, self).__init__()
        self.__subgraph_num = subgraph_num
        self.__hop_num = hop_num
        self.__feat_dim = feat_dim

        self.__learnable_weight = nn.ParameterList()
        for _ in range(hop_num):
            self.__learnable_weight.append(nn.Parameter(torch.FloatTensor(feat_dim, subgraph_num)))

    # feat_list_list = hop_num * feat_list = hop_num * (subgraph_num * feat)
    def forward(self, feat_list_list):
        aggregated_feat_list = []
        for i in range(self.__hop_num):
            adopted_feat = torch.stack(feat_list_list[i], dim=2)
            intermediate_feat = (adopted_feat * (self.__learnable_weight[i].unsqueeze(dim=0))).mean(dim=2)

            aggregated_feat_list.append(intermediate_feat)

        return aggregated_feat_list


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.__fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feature):
        output = self.__fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bn=False):
        super(MultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.__num_layers = num_layers

        self.__fcs = nn.ModuleList()
        self.__fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.__fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.__fcs.append(nn.Linear(hidden_dim, num_classes))

        self.__bn = bn
        if self.__bn is True:
            self.__bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.__bns.append(nn.BatchNorm1d(hidden_dim))

        self.__dropout = nn.Dropout(dropout)

    def forward(self, feature):
        for i in range(self.__num_layers - 1):
            feature = self.__fcs[i](feature)
            if self.__bn is True:
                feature = self.__bns[i](feature)
            feature = F.relu(feature)
            feature = self.__dropout(feature)

        output = self.__fcs[-1](feature)
        return output
