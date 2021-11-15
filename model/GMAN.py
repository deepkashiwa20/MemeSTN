import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
class spatialAttention(nn.Module):
    """
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d 
        self.d = d 
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    """
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    """
    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d 
        self.d = d 
        self.K = K 
        self.mask = mask 
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        
    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -1 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X
    

class gatedFusion(nn.Module):
    """
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    """

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                        bn_decay=bn_decay)
    
    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    """
    transform attention mechanism
    X:          [batch_size, num_his, num_vertex, D]
    STE_his:    [batch_size, num_his, num_vertex, D]
    STE_pred:   [batch_size, num_pred, num_vertex, D]
    K:          number of attention heads
    d:          dimension of each attention outputs
    return:     [batch_size, num_pred, num_vertex, D]
    """

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d 
        self.K = K
        self.d = d 
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class STEmbedding(nn.Module):
    """
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_hist + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    return: [batch_size, num_his + num_pred, num_vertex, D]
    """

    def __init__(self, SE_dim, TE_dim, D, bn_decay, device):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[SE_dim, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )

        self.FC_te = FC(
            input_dims=[TE_dim, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )
        self.device = device

    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        TE = TE.unsqueeze(dim=2).to(device=self.device)
        TE = self.FC_te(TE)
        return SE + TE
    
class GMAN(nn.Module):
    """
    GMAN
        X:          [batch_size, num_his, num_vertex]
        TE:         [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE:         [num_vertex, K * d]
        num_his:    number of history steps
        num_pred:   number of prediction steps
        T:          one day is divided into T steps
        L:          number of STAtt blocks in the encoder/decoder
        K:          number of attention heads
        d:          dimension of each attention head outputs
        return:     [batch_size, num_pred, num_vertex]
    """

    def __init__(self, SE, SE_dim, TE_dim, timestep_in, channel_in, channel_out, device, statt_layers=1, att_heads=8, att_dims=8, bn_decay=0.1):
        super(GMAN, self).__init__()
        L = statt_layers 
        K = att_heads
        d = att_dims
        D = K * d 
        self.num_his = timestep_in
        self.SE = SE 
        self.STEmbedding = STEmbedding(SE_dim, TE_dim, D, bn_decay, device)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[channel_in, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, channel_out], activations=[F.relu, None],
                       bn_decay=bn_decay)
        
    def forward(self, X, TE):
        # input
        X = self.FC_1(X)
        # STE
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return X


def main():
    # BATCHSIZE = 16
    SE_DIM = 64 # GMAN spatial embedding(adjacent matrix embedding) dim
    TE_DIM = 32
    SEPATH = '../PEMSBAY/SE_pemsbay.txt'
    TIMESTEP_IN = 12
    TIMESTEP_OUT = 12 # It can be overrided by param_stgcn3
    N_NODE = 47
    CHANNEL = 1
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    SE = torch.zeros((N_NODE, SE_DIM), dtype=torch.float32).to(device=device)
    model = GMAN(SE, SE_DIM, TE_DIM, TIMESTEP_IN, device).to(device)
    summary(model, [(TIMESTEP_IN, N_NODE), (TIMESTEP_IN+TIMESTEP_OUT, TE_DIM)], device=device)
    
if __name__ == '__main__':
    main()