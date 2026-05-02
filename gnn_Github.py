import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def gmul(input):
    edge_weight, node_features = input
    edge_weight_size = edge_weight.size()
    N = edge_weight_size[-2]
    edge_weight = edge_weight.split(1, 3)
    edge_weight = torch.cat(edge_weight, 1).squeeze(3)
    output = torch.bmm(edge_weight, node_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)
    return output

class Gconv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, ratio=2, bn_bool=True):
        super(Gconv, self).__init__()
        self.ratio = ratio
        self.in_channels = ratio * in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.out_channels)
    def forward(self, input):
        edge_weight = input[0]
        node_features = gmul(input)
        if self.ratio == 1:
            node_features = torch.abs(node_features)
        node_features_size = node_features.size()
        node_features = node_features.contiguous()
        node_features = node_features.view(-1, self.in_channels)
        node_features = self.dropout(self.fc(node_features))
        if self.bn_bool:
            node_features = self.bn(node_features)
        node_features = node_features.view(*node_features_size[:-1], self.out_channels)
        return edge_weight, node_features

class Wcompute(nn.Module):
    def __init__(self, in_channels, out_channels, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], dropout=0.5):
        super(Wcompute, self).__init__()
        self.out_channels = out_channels
        self.operator = operator
        self.activation = activation
        self.conv2d_1 = nn.Conv2d(in_channels, int(out_channels * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(out_channels * ratio[0]))
        self.dropout = nn.Dropout(dropout)
        self.conv2d_2 = nn.Conv2d(int(out_channels * ratio[0]), int(out_channels * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(out_channels * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(out_channels * ratio[1]), int(out_channels * ratio[2]), 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(int(out_channels * ratio[2]))
        self.conv2d_4 = nn.Conv2d(int(out_channels * ratio[2]), int(out_channels * ratio[3]), 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(int(out_channels * ratio[3]))
        self.conv2d_last = nn.Conv2d(out_channels, 1, 1, stride=1)
    def forward(self, x, W_id): # x:(210,8,48) W_id:(210,8,8,1)
        W1 = x.unsqueeze(2) # (210,8,1,48)
        W2 = torch.transpose(W1, 1, 2) # (210,1,8,48)
        W_new = torch.abs(W1 - W2) # (210,8,8,48)
        W_new = torch.transpose(W_new, 1, 3) # (210,48,8,8)
        W_new = self.conv2d_1(W_new) # (210,48,8,8)
        W_new = self.bn_1(W_new) # (210,48,8,8)
        W_new = F.leaky_relu(W_new) # (210,48,8,8)
        W_new = self.dropout(W_new) # (210,48,8,8)
        W_new = self.conv2d_2(W_new) # (210,48,8,8)
        W_new = self.bn_2(W_new) # (210,48,8,8)
        W_new = F.leaky_relu(W_new) # (210,48,8,8)
        W_new = self.conv2d_3(W_new) # (210,48,8,8)
        W_new = self.bn_3(W_new) # (210,48,8,8)
        W_new = F.leaky_relu(W_new) # (210,48,8,8)
        W_new = self.conv2d_4(W_new) # (210,48,8,8)
        W_new = self.bn_4(W_new) # (210,48,8,8)
        W_new = F.leaky_relu(W_new) # (210,48,8,8)
        W_new = self.conv2d_last(W_new) # (210,1,8,8)
        W_new = torch.transpose(W_new, 1, 3) # (210,8,8,1)
        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8 # (210,8,8,1)
            W_new = torch.transpose(W_new, 2, 3) # (210,8,1,8)
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim=-1)
            W_new = W_new.view(W_new_size)
            W_new = torch.transpose(W_new, 2, 3)
        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)
        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)
        return W_new

class GNN(nn.Module):
    def __init__(self, in_channel, out_channel, G_ratio=2, W_ratio=[2, 1.5, 1, 1], num_layers=2, dropout=0.5):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            module_w = Wcompute(in_channel + int(out_channel // 2) * i,
                                in_channel + int(out_channel // 2) * i,
                                operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
            module_l = Gconv(in_channel + int(out_channel // 2) * i, int(out_channel // 2), dropout=dropout, ratio=G_ratio, bn_bool=True)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)
        self.w_comp_last = Wcompute(in_channel + int(out_channel // 2) * self.num_layers,
                                    in_channel + int(out_channel // 2) * (self.num_layers - 1),
                                    operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        self.layer_last = Gconv(in_channel + int(out_channel // 2) * self.num_layers, in_channel, dropout=dropout, ratio=G_ratio, bn_bool=True)
    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if torch.cuda.is_available():
            W_init = W_init.cuda()
        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)
        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]
        return out