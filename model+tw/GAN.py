import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import math
import struct
import copy
import sys
from torchsummary import summary

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DyConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(DyConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class Building_Block(nn.Module):
    def __init__(self, hidden_features, num_heads, dropout):
        super(Building_Block, self).__init__()
        self.embed_dim = hidden_features
        
        self.attn = MultiHeadAttention(num_heads, self.embed_dim, dropout)
        self.pos_ffn = FeedForward(self.embed_dim, dropout=dropout)

        self.norm_1 = Norm(self.embed_dim)
        self.norm_2 = Norm(self.embed_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):  # here x includes c
        # input x <- (batch_size, sequence_length, pixel_num * feature_num)
        x2 = self.norm_1(x)
        new_x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(new_x)
        new_x = new_x + self.dropout_2(self.pos_ffn(x2))
        # output new_x <- (batch_size, sequence_length, pixel_num * feature_num)
        return new_x
    
class Generator(nn.Module):
    # final_features is the num of features we want (usually is 1)
    def __init__(self, init_features, cond_features, hidden_features, final_features, num_heads, dropout, num_block, num_variable, seq_len):
        super(Generator, self).__init__()
        self.num_block = num_block
        self.seq_len = seq_len
        self.num_variable = num_variable 
        # Building_Block(hidden_features, num_heads, dropout)
        # Building_Block: hidden_features = num_variable * final_features 
        self.linear = nn.Linear(final_features, final_features)
        self.blocks = get_clones(Building_Block(num_variable * final_features, num_heads, dropout), self.num_block)
        self.norm = Norm(num_variable * final_features)
            
        self.DyConv1 = DyConv(init_features + cond_features, hidden_features)
        self.DyConv2 = DyConv(hidden_features, int(hidden_features / 2))
        self.DyConv3 = DyConv(int(hidden_features / 2), int(hidden_features / 4))
        self.DyConv4 = DyConv(int(hidden_features / 4), final_features)

        self.bn1 = nn.BatchNorm1d(num_variable * final_features)
        self.bn2 = nn.BatchNorm1d(num_variable * final_features)
        self.bn3 = nn.BatchNorm1d(num_variable * final_features)
        self.bn4 = nn.BatchNorm1d(num_variable * final_features)
        
    def forward(self, x, adj, c):
        # input noise x <- (batch_size, sequence_length, pixel_num, init_features = 100 rand nums)
        # input adjacency matrix <- (batch_size, sequence_length, pixel_num, pixel_num)
        # input condition c <- (batch_size, sequence_length, pixel_num, condition_num)

        x = torch.cat((x, c), dim=3)

        # first x go through one GraphConvolution layer
        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.relu(self.bn1(self.DyConv1(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.relu(self.bn2(self.DyConv2(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.relu(self.bn3(self.DyConv3(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.relu(self.bn4(self.DyConv4(x[:, i, :, :], adj[:, i, :, :]))))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)
        x = x.view(x.size()[0], x.size()[1], -1)

        # second, x go through many times building_blocks
        for i in range(self.num_block):
            x = self.blocks[i](x)
        '''
        # x shape: (batch_size, sequence_length, pixel_num * hidden_features)
        x = self.Building_Block(x, adj)
        '''
        x = self.norm(x)
        x = x.view(x.size()[0], x.size()[1], self.num_variable, -1)

        # x <- (batch_size, sequence_length, pixel_num, final_features)
        x = torch.tanh(self.linear(x))

        return x


class Discriminator(nn.Module):
    # final_features usually is 1, cuz we need one scalar
    def __init__(self, init_features, cond_features, hidden_features, final_features, num_heads, dropout, num_block, num_variable, seq_len):
        super(Discriminator, self).__init__()
        self.num_block = num_block
        self.seq_len = seq_len
        # Building_Block(hidden_features, num_heads, dropout)
        # Building_Block: hidden_features = num_variable * final_features 
        self.linear = nn.Linear(num_variable * final_features, final_features)
        self.blocks = get_clones(Building_Block(num_variable * final_features, num_heads, dropout), self.num_block) # should be equal to DyConv4 output
        self.norm = Norm(num_variable * final_features)
            
        self.DyConv1 = DyConv(init_features + cond_features, hidden_features)
        self.DyConv2 = DyConv(hidden_features, int(hidden_features * 2))
        self.DyConv3 = DyConv(int(hidden_features * 2), int(hidden_features * 4))
        self.DyConv4 = DyConv(int(hidden_features * 4), final_features)
        
        self.bn1 = nn.BatchNorm1d(num_variable * final_features)
        self.bn2 = nn.BatchNorm1d(num_variable * final_features)
        self.bn3 = nn.BatchNorm1d(num_variable * final_features)
        self.bn4 = nn.BatchNorm1d(num_variable * final_features)
        
    def forward(self, x, adj, c):
        # input region x <- (batch_size, sequence_length, pixel_num, init_feature_num)
        # input adjacency matrix <- (batch_size, sequence_length, pixel_num, pixel_num)
        # input condition c <- (batch_size, sequence_length, pixel_num, condition_num)
            
        x = torch.cat((x, c), dim=3)

        # first x go through one GraphConvolution layer
        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.leaky_relu(self.bn1(self.DyConv1(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.leaky_relu(self.bn2(self.DyConv2(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.leaky_relu(self.bn3(self.DyConv3(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)

        new_x = []
        for i in range(self.seq_len):
            new_x.append(F.leaky_relu(self.bn4(self.DyConv4(x[:, i, :, :], adj[:, i, :, :])), 0.2))

        new_x = torch.stack(new_x)
        # x <- (batch_size, sequence_length, pixel_num, feature_num)
        x = new_x.permute(1, 0, 2, 3)
        x = x.view(x.size()[0], x.size()[1], -1)

        # second, x go through many times building_blocks
        for i in range(self.num_block):
            x = self.blocks[i](x)

        x = self.norm(x)

        '''
        # x shape: (batch_size, sequence_length, pixel_num * hidden_features)
        x = self.Building_Block(x, adj)
        '''
        # outputs <- (batch_size, sequence_length, 1)
        outputs = torch.sigmoid(self.linear(x))
        # outputs <- (batch_size, 1)
        outputs = torch.mean(outputs, dim=1, keepdim=False)

        return outputs

def main():
    dropout = 0.1
    num_head = 7
    num_block_D = 2
    num_block_G = 3
    num_variable = 7 # How many variables we want to predict, original pix_num 10*10=100.
    final_feat = 1
    seq_len = 12
    D_init_feat = 1
    G_init_feat = 100
    D_hidden_feat = 16
    G_hidden_feat = 64
    cond_feat = 32
    # cond_feat = 34
    # cond_sources = 2
    
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '0'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

    D = Discriminator(D_init_feat, cond_feat, D_hidden_feat, final_feat, num_head, dropout, num_block_D, num_variable, seq_len).to(device)
    G = Generator(G_init_feat, cond_feat, G_hidden_feat, final_feat, num_head, dropout, num_block_G, num_variable, seq_len).to(device)
    adj = (seq_len, num_variable, num_variable)
    c = (seq_len, num_variable, cond_feat)
    D_x = (seq_len, num_variable, D_init_feat)
    G_x = (seq_len, num_variable, G_init_feat)
    
    print('########## This is the model summary of Descriminator ############')
    # summary(D, [D_x, adj, c])
    print('########## This is the model summary of Generator ############')
    summary(G, [G_x, adj, c])
    
if __name__ == '__main__':
    main()