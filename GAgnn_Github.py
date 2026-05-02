import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def make_G4_mul(kernel):
    dim = kernel.size(1) // 4
    w0, w1, w2, w3 = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    ze = torch.zeros_like(w0, device=kernel.device, requires_grad=False)
    r = torch.cat([w0, w1, w2, w3], dim=0)
    i = torch.cat([w1, w0, ze, ze], dim=0)
    j = torch.cat([w2, ze, w0, ze], dim=0)
    k = torch.cat([w1, ze, ze, w0], dim=0)
    l = torch.cat([ze, w2, -w1, ze], dim=0)
    m = torch.cat([ze, ze, w3, -w2], dim=0)
    n = torch.cat([ze, -w3, ze, w1], dim=0)
    hamilton = torch.cat([r, i, j, k, l, m, n], dim=1)
    return hamilton

def make_G8_mul(kernel):
    dim = kernel.size(1) // 8
    w0, w1, w2, w3, w4, w5, w6, w7 = torch.split(kernel, [dim, dim, dim, dim, dim, dim, dim, dim], dim=1)
    w02 = torch.cat([w0, w1, w2, w3, -w4, -w5, -w6, -w7], dim=0)
    w12 = torch.cat([w1, w0, -w4, w6, w2, -w7, -w3, -w5], dim=0)
    w22 = torch.cat([w2, w4, w0, w5, -w1, -w3, -w7, -w6], dim=0)
    w32 = torch.cat([w3, -w6, w5, w0, -w7, -w2, w1, -w4], dim=0)
    w42 = torch.cat([w4, w2, -w1, w7, w0, -w6, w5, w3], dim=0)
    w52 = torch.cat([w5, w7, w3, -w2, w6, w0, -w4, w1], dim=0)
    w62 = torch.cat([w6, -w3, w7, w1, -w5, w4, w0, w2], dim=0)
    w72 = torch.cat([w7, w5, w6, w4, w3, w1, w2, w0], dim=0)
    hamilton = torch.cat([w02, w12, w22, w32, w42, w52, w62, w72], dim=1)
    return hamilton

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
    def __init__(self, in_channels, out_channels, dropout, ratio=2, bn_bool=True, mode='Vanilla-GCN'):
        super(Gconv, self).__init__()
        self.ratio = ratio
        self.in_channels = ratio * in_channels
        self.out_channels = out_channels
        self.mode = mode
        if self.mode == 'Vanilla-GCN':
            self.weight = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        elif self.mode == 'G4-GCN':
            self.weight = Parameter(torch.FloatTensor(self.in_channels // 4, self.out_channels // 7 * 4))
        elif self.mode == 'G8-GCN':
            self.weight = Parameter(torch.FloatTensor(self.in_channels // 8, self.out_channels))
        else:
            sys.exit()
        self.dropout = nn.Dropout(dropout)
        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.out_channels)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        edge_weight = input[0]
        node_features = gmul(input)
        if self.ratio == 1:
            node_features = torch.abs(node_features)
        node_features_size = node_features.size()
        node_features = node_features.contiguous() # (210,8,96)
        node_features = node_features.view(-1, self.in_channels) # (1680,96)
        if self.mode == 'Vanilla-GCN':
            node_features = self.dropout(torch.mm(node_features, self.weight))
        elif self.mode == 'G4-GCN':
            hamilton = make_G4_mul(self.weight)
            node_features = self.dropout(torch.mm(node_features, hamilton))
        elif self.mode == 'G8-GCN':
            hamilton = make_G8_mul(self.weight)
            node_features = self.dropout(torch.mm(node_features, hamilton))
        if self.bn_bool:
            node_features = self.bn(node_features)
        node_features = node_features.view(*node_features_size[:-1], self.out_channels)
        return edge_weight, node_features

class Wcompute(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, operator='J2', activation='softmax', ratio=[2, 2, 1, 1]):
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
    def __init__(self, in_channel, out_channel, G8_channel=56, G_ratio=2, W_ratio=[2, 1.5, 1, 1], dropout=0.5):
        super(GNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.G8_channel = G8_channel
        module_w = Wcompute(in_channel, in_channel, operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        module_g = Gconv(in_channel, int(out_channel // 2), dropout=dropout, ratio=G_ratio, bn_bool=True, mode='Vanilla-GCN')
        self.add_module('layer_w0', module_w)
        self.add_module('layer_g0', module_g)

        self.fc_1 = nn.Linear(in_channel + int(out_channel // 2), G8_channel)
        self.dropout = nn.Dropout(dropout)
        self.bn_1 = nn.BatchNorm1d(G8_channel)
        module_w = Wcompute(G8_channel, G8_channel, operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        module_g = Gconv(G8_channel, G8_channel, dropout=dropout, ratio=G_ratio, bn_bool=True, mode='G8-GCN')
        self.add_module('layer_w1', module_w)
        self.add_module('layer_g1', module_g)

        module_w = Wcompute(G8_channel // 2, G8_channel // 2, operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        module_g = Gconv(G8_channel // 2, G8_channel // 2, dropout=dropout, ratio=G_ratio, bn_bool=True, mode='G4-GCN')
        self.add_module('layer_w2_1', module_w)
        self.add_module('layer_g2_1', module_g)
        module_w = Wcompute(G8_channel // 2, G8_channel // 2, operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        module_g = Gconv(G8_channel // 2, G8_channel // 2, dropout=dropout, ratio=G_ratio, bn_bool=True, mode='G4-GCN')
        self.add_module('layer_w2_2', module_w)
        self.add_module('layer_g2_2', module_g)
        self.fc_2 = nn.Linear(G8_channel, in_channel + int(out_channel // 2))
        self.bn_2 = nn.BatchNorm1d(in_channel + int(out_channel // 2))

        module_w = Wcompute(in_channel + int(out_channel // 2), in_channel + int(out_channel // 2), operator='J2', activation='softmax', ratio=W_ratio, dropout=dropout)
        module_g = Gconv(in_channel + int(out_channel // 2), in_channel, dropout=dropout, ratio=G_ratio, bn_bool=True, mode='Vanilla-GCN')
        self.add_module('layer_w3', module_w)
        self.add_module('layer_g3', module_g)
    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if torch.cuda.is_available():
            W_init = W_init.cuda()
        Wi = self._modules['layer_w0'](x, W_init)
        x_new = F.leaky_relu(self._modules['layer_g0']([Wi, x])[1])
        x = torch.cat([x, x_new], dim=2)
        identity1 = x

        x = self.bn_1(self.dropout(self.fc_1(x.view(-1, self.in_channel + int(self.out_channel // 2) * 0 + int(self.out_channel // 2)))))
        x = x.view(*identity1.size()[:-1], self.G8_channel)
        identity2 = x
        Wi = self._modules['layer_w1'](x, W_init)
        x = F.leaky_relu(self._modules['layer_g1']([Wi, x])[1])
        x1, x2 = torch.split(x, [self.G8_channel // 2, self.G8_channel // 2], dim=2)

        Wi = self._modules['layer_w2_1'](x1, W_init)
        x1 = F.leaky_relu(self._modules['layer_g2_1']([Wi, x1])[1])
        Wi = self._modules['layer_w2_2'](x2, W_init)
        x2 = F.leaky_relu(self._modules['layer_g2_2']([Wi, x2])[1])
        out = torch.cat([x1, x2], dim=2)

        out += identity2
        out = self.bn_2(self.dropout(self.fc_2(out.view(-1, self.G8_channel))))
        out = out.view(*identity2.size()[:-1], self.in_channel + int(self.out_channel // 2))

        out += identity1
        Wi = self._modules['layer_w3'](out, W_init)
        out = F.leaky_relu(self._modules['layer_g3']([Wi, out])[1])
        return out