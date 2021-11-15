import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


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

    def __init__(self, K, d, m, bn_decay, if_STE):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.if_STE = if_STE
        if self.if_STE:
            self.FC_q = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_k = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_v = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC = FC(input_dims=D, units=D, activations=F.relu,
                         bn_decay=bn_decay)
        else:
            self.FC_q = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_k = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_v = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC = FC(input_dims=D, units=D, activations=F.relu,
                         bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        if (self.if_STE)&(STE is not None):
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

    def __init__(self, K, d, m, bn_decay, if_STE, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.if_STE = if_STE
        if self.if_STE:
            self.FC_q = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_k = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_v = FC(input_dims=2 * D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC = FC(input_dims=D, units=D, activations=F.relu,
                         bn_decay=bn_decay)
        else:
            self.FC_q = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_k = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC_v = FC(input_dims=D + m, units=D, activations=F.relu,
                           bn_decay=bn_decay)
            self.FC = FC(input_dims=D, units=D, activations=F.relu,
                         bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        if (self.if_STE)&(STE is not None):
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

    def __init__(self, D, m, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D+m], activations=[F.relu, None],
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
    def __init__(self, K, d, m, bn_decay, if_STE, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, m, bn_decay, if_STE)
        self.temporalAttention = temporalAttention(K, d, m, bn_decay, if_STE, mask=mask)
        self.gatedFusion = gatedFusion(K * d, m, bn_decay)

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

    def __init__(self, K, d, m, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D+m, units=D, activations=F.relu,
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
        return SE + TE      # match dim and sum


class GMAN_GloMem(nn.Module):
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
    def __init__(self, SE, N, SE_dim, TE_dim, timestep_in, timestep_out, device, statt_layers=1, att_heads=8, att_dims=8, bn_decay=0.1,
                 mem_num:int=5, mem_dim:int=8):
        super(GMAN_GloMem, self).__init__()
        L = statt_layers
        K = att_heads
        d = att_dims
        D = K * d
        self.num_his = timestep_in
        self.num_pred = timestep_out
        self.SE = SE
        self.STEmbedding = STEmbedding(SE_dim, TE_dim, D, bn_decay, device)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, 0, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, mem_dim, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, 0, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D+mem_dim, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # global memory use
        self.N = N
        self.D = D
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        #self.st_memory = self.construct_st_memory()
        self.seq_memory = self.construct_seq_memory()

    def construct_st_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.N*self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim*self.N), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        h_t_flat = h_t.reshape(h_t.shape[0], -1)    # (B, N*h)
        query = torch.mm(h_t_flat, self.st_memory['Wa'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.st_memory['memory'].t()), dim=1)         # alpha: (B, M)
        mem_t = torch.mm(torch.mm(att_score, self.st_memory['memory']), self.st_memory['fc'])   # (B, N*d)
        _h_t = torch.cat([h_t, mem_t.reshape(h_t.shape[0], self.N, self.mem_dim)], dim=-1)      # (B, N, h+d)
        return _h_t

    def construct_seq_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.num_pred*self.N*self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim*self.N*self.num_pred), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_seq_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 4, 'Input to query Seq-Memory must be a 4D tensor'

        h_t_flat = h_t.reshape(h_t.shape[0], -1)    # (B, T*N*h)
        query = torch.mm(h_t_flat, self.seq_memory['Wa'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.seq_memory['memory'].t()), dim=1)         # alpha: (B, M)
        mem_t = torch.mm(torch.mm(att_score, self.seq_memory['memory']), self.seq_memory['fc'])   # (B, N*d)
        _h_t = torch.cat([h_t, mem_t.reshape(h_t.shape[0], self.num_pred, self.N, self.mem_dim)], dim=-1)      # (B, N, h+d)
        return _h_t

    def forward(self, X, TE):
        # 2 branches: 0 mobility, 1 twitter
        X, TW = X[..., 0], X[..., -1]
        # input
        X = torch.unsqueeze(X, -1)
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
        # query seq memory
        X = self.query_seq_memory(X)
        # query ST memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X = torch.stack(x, dim=1)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # query ST memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X = torch.stack(x, dim=1)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, -1)


class GMAN_LocMem(nn.Module):
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
    def __init__(self, SE, N, SE_dim, TE_dim, timestep_in, timestep_out, device, statt_layers=1, att_heads=8, att_dims=8, bn_decay=0.1,
                 mem_num:int=5, mem_dim:int=8):
        super(GMAN_LocMem, self).__init__()
        L = statt_layers
        K = att_heads
        d = att_dims
        D = K * d
        self.num_his = timestep_in
        self.num_pred = timestep_out
        self.SE = SE
        self.STEmbedding = STEmbedding(SE_dim, TE_dim, D, bn_decay, device)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, 0, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, mem_dim, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, 0, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D+mem_dim, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)
        # local memory use
        self.N = N
        self.D = D
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.st_memory = self.construct_st_memory()

    def construct_st_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.N, self.mem_num, self.mem_dim), requires_grad=True)     # (N, M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        query = torch.einsum('bnh,hd->bnd', h_t, self.st_memory['Wa'])      # (B, N, d)
        att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, self.st_memory['memory']), dim=-1)  # alpha: (B, N, M)
        proto_t = torch.einsum('bnm,nmd->bnd', att_score, self.st_memory['memory'])      # (B, N, d)
        mem_t = torch.matmul(proto_t, self.st_memory['fc'])     # (B, N, d)

        _h_t = torch.cat([h_t, mem_t], dim=-1)      # (B, N, h+d)
        return _h_t

    def forward(self, X, TE):
        # 2 branches: 0 mobility, 1 twitter
        X, TW = X[...,0], X[...,-1]
        # input
        X = torch.unsqueeze(X, -1)
        X = self.FC_1(X)
        # STE
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # query ST memory
        # x = []
        # for t in range(self.num_his):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X = torch.stack(x, dim=1)
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # query ST memory
        x = []
        for t in range(self.num_pred):
            x.append(self.query_st_memory(X[:,t,:,:]))
        X = torch.stack(x, dim=1)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # query ST memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X = torch.stack(x, dim=1)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, -1)


class GMAN_CluMem(nn.Module):
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
    def __init__(self, SE, N, SE_dim, TE_dim, timestep_in, timestep_out, device, statt_layers=1, att_heads=8, att_dims=8, bn_decay=0.1,
                 num_cluster:int=128, mem_num:int=5, mem_dim:int=8):
        super(GMAN_CluMem, self).__init__()
        L = statt_layers
        K = att_heads
        d = att_dims
        D = K * d
        self.num_his = timestep_in
        self.num_pred = timestep_out
        self.SE = SE
        self.STEmbedding = STEmbedding(SE_dim, TE_dim, D, bn_decay, device)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, 0, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, mem_dim, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, 0, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D+mem_dim, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        # clustered memory use
        self.SE_dim = SE_dim
        self.k = num_cluster
        self.D = D
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.st_memory = self.construct_st_memory()

    def construct_st_memory(self):
        memory_weight = nn.ParameterDict()
        # memory_weight['SE_proj'] = nn.Parameter(torch.randn(self.SE_dim, self.k), requires_grad=True)   # (SE, k)
        # nn.init.xavier_normal_(memory_weight['SE_proj'])
        # memory_weight['mem_pool'] = nn.Parameter(torch.randn(self.k, self.mem_num, self.mem_dim), requires_grad=True)     # (k, M, d)
        # nn.init.xavier_normal_(memory_weight['mem_pool'])
        memory_weight['mem_pool'] = nn.Parameter(torch.randn(self.SE_dim, self.mem_num, self.mem_dim), requires_grad=True)     # (SE, M, d)
        nn.init.xavier_normal_(memory_weight['mem_pool'])
        # memory_weight['pool1'] = nn.Parameter(torch.randn(self.mem_num, self.SE_dim), requires_grad=True)     # (M, SE)
        # memory_weight['pool2'] = nn.Parameter(torch.randn(self.mem_dim, self.SE_dim), requires_grad=True)     # (d, SE)
        # nn.init.xavier_normal_(memory_weight['pool1'])
        # nn.init.xavier_normal_(memory_weight['pool2'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        # recon_mem = torch.einsum('ne,ek,kmd->nmd', self.SE, self.st_memory['SE_proj'], self.st_memory['mem_pool'])  # (N, M, d)
        recon_mem = torch.einsum('ne,emd->nmd', self.SE, self.st_memory['mem_pool'])    # (N, M, d)
        # recon_mem = torch.einsum('ne,me,de->nmd', self.SE, self.st_memory['pool1'], self.st_memory['pool2'])    # (N, M, d)
        query = torch.einsum('bnh,hd->bnd', h_t, self.st_memory['Wa'])      # (B, N, d)
        att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, recon_mem), dim=-1)  # alpha: (B, N, M)
        proto_t = torch.einsum('bnm,nmd->bnd', att_score, recon_mem)      # (B, N, d)
        mem_t = torch.matmul(proto_t, self.st_memory['fc'])     # (B, N, d)

        _h_t = torch.cat([h_t, mem_t], dim=-1)      # (B, N, h+d)
        return _h_t

    def forward(self, X, TE):
        # 2 branches: 0 mobility, 1 twitter
        X, TW = X[..., 0], X[..., -1]
        # input
        X = torch.unsqueeze(X, -1)
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
        # query ST memory
        x = []
        for t in range(self.num_pred):
            x.append(self.query_st_memory(X[:,t,:,:]))
        X = torch.stack(x, dim=1)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # query ST memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X = torch.stack(x, dim=1)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, -1)


class MetaSTEmbedding(nn.Module):
    """
    meta spatio-temporal embedding with distilled twitter knowledge
    TW:     [batch_size, num_hist, num_vertex, D]
    SE:     [num_vertex, D]
    TE:     [batch_size, num_hist + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    return: [batch_size, num_his + num_pred, num_vertex, D]
    """

    def __init__(self, N, SE_dim, TE_dim, D, bn_decay, device, mem_num:int=5, mem_dim:int=8):
        super(MetaSTEmbedding, self).__init__()
        self.N = N
        self.D = D
        self.SE_dim = SE_dim
        self.TE_dim = TE_dim
        self.device = device
        self.tw2se_W = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, SE_dim*D))    # todo: + BN
        self.tw2se_b = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.tw2te_W = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, TE_dim*D))
        self.tw2te_b = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        # memory use
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.global_memory = self.construct_global_memory()
        # self.local_memory = self.construct_local_memory()
        # memory fusion
        self.mem_fus = GLU(D, D)

    def construct_global_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.N*self.D, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['FC_SE'] = nn.Parameter(torch.randn(self.mem_dim, self.SE_dim*self.D), requires_grad=True)
        memory_dict['FC_TE'] = nn.Parameter(torch.randn(self.mem_dim, self.TE_dim*self.D), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_global_memory(self, TW_t:Tensor):
        assert len(TW_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'
        B = TW_t.shape[0]

        TW_t_flat = TW_t.reshape(B, -1)
        query = torch.mm(TW_t_flat, self.global_memory['Wq'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.global_memory['Memory'].t()), dim=1)         # alpha: (B, M)
        proto_t = torch.mm(att_score, self.global_memory['Memory'])     # (B, d)

        Wse = torch.matmul(proto_t, self.global_memory['FC_SE'])    # (B, emb*d)
        Wte = torch.matmul(proto_t, self.global_memory['FC_TE'])    # (B, emb*d)
        Wse, Wte = Wse.reshape(B, self.SE_dim, self.D), Wte.reshape(B, self.TE_dim, self.D)
        return Wse, Wte, query, att_score

    def get_NP_pair(self, att_score:Tensor):
        _, ind = torch.topk(att_score, k=2, dim=1)
        pos = self.global_memory['Memory'][ind[:,0]]
        neg = self.global_memory['Memory'][ind[:,1]]
        # print(pos.shape, neg.shape)
        return pos, neg

    # def construct_local_memory(self):
    #     memory_dict = nn.ParameterDict()
    #     memory_dict['Memory'] = nn.Parameter(torch.randn(self.N, self.mem_num, self.mem_dim), requires_grad=True)     # (N, M, d)
    #     memory_dict['Wq'] = nn.Parameter(torch.randn(self.D, self.mem_dim), requires_grad=True)    # project to query
    #     memory_dict['FC_SE'] = nn.Parameter(torch.randn(self.mem_dim, self.SE_dim*self.D), requires_grad=True)
    #     memory_dict['FC_TE'] = nn.Parameter(torch.randn(self.mem_dim, self.TE_dim*self.D), requires_grad=True)
    #     for param in memory_dict.values():
    #         nn.init.xavier_normal_(param)
    #     return memory_dict
    #
    # def query_local_memory(self, TW_t:Tensor):
    #     assert len(TW_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'
    #     B = TW_t.shape[0]
    #
    #     query = torch.einsum('bnh,hd->bnd', TW_t, self.local_memory['Wq'])     # (B, N, d)
    #     att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, self.local_memory['Memory']), dim=-1)    # alpha: (B, N, M)
    #     proto_t = torch.einsum('bnm,nmd->bnd', att_score, self.local_memory['Memory'])     # (B, N, d)
    #
    #     Wse = torch.matmul(proto_t, self.local_memory['FC_SE'])    # (B, N, emb*d)
    #     Wte = torch.matmul(proto_t, self.local_memory['FC_TE'])    # (B, N, emb*d)
    #     Wse, Wte = Wse.reshape(B, self.N, self.SE_dim, self.D), Wte.reshape(B, self.N, self.TE_dim, self.D)
    #     return Wse, Wte, query, att_score

    # def generate_NP_pairs(self, att_score:Tensor):      # w/ tw memory
    #     assert len(att_score.shape) == 3
    #
    #     val, ind = att_score.sort(descending=True, dim=-1)
    #     pos, neg = [], []
    #     for b in range(ind.shape[0]):
    #         b_pos, b_neg = [], []
    #         for n in range(ind.shape[1]):
    #             n_neg = []
    #             for i in range(ind.shape[2]):
    #                 if i == 0:      # most similar as positive
    #                     b_pos.append(self.tw_memory['Memory'][n,ind[b,n,i],:])
    #                 else:
    #                     n_neg.append(self.tw_memory['Memory'][n,ind[b,n,i],:])
    #             b_neg.append(torch.stack(n_neg, dim=-1))
    #         pos.append(torch.stack(b_pos, dim=0))
    #         neg.append(torch.stack(b_neg, dim=0))
    #     pos = torch.stack(pos, dim=0)   # (B, N, d)
    #     neg = torch.stack(neg, dim=0)   # (B, N, d, M-1)
    #     return pos, neg

    def generate_NP_pairs(self, att_score:Tensor):      # w/ tw memory
        assert len(att_score.shape) == 2

        val, ind = att_score.sort(descending=True, dim=1)       # topk more efficient!
        pos, neg = [], []
        for b in range(ind.shape[0]):
            b_neg = []
            for i in range(ind.shape[1]):
                if i == 0:      # most similar
                    pos.append(self.tw_memory['Memory'][ind[b,i],:])
                else:
                    b_neg.append(self.tw_memory['Memory'][ind[b,i],:])
            neg.append(torch.stack(b_neg, dim=1))
        pos = torch.stack(pos, dim=0)   # (B, d)
        neg = torch.stack(neg, dim=0)   # (B, d, M-1)
        return pos, neg

    def forward(self, TW, SE, TE):      # direct tweet_seq input
        TW_t = TW[:,-1]   # last hidden state
        # B, N, _ = TW.shape
        # se_W, se_b = self.tw2se_W(TW).reshape(B, N, self.SE_dim, self.D), self.tw2se_b(TW)
        # metaSE = torch.einsum('ne,bned->bnd', SE, se_W) + se_b
        # te_W, te_b = self.tw2te_W(TW).reshape(B, N, self.TE_dim, self.D), self.tw2te_b(TW).unsqueeze(1)
        # metaTE = torch.einsum('bte,bned->btnd', TE.to(device=self.device), te_W) + te_b

        # global
        Wse, Wte, query, att_score = self.query_global_memory(TW_t)
        pos, neg = self.get_NP_pair(att_score)
        metaSE = torch.einsum('ne,bed->bnd', SE, Wse)
        metaTE = torch.einsum('bte,bed->btd', TE.to(device=self.device), Wte)
        global_metaSTE = metaSE.unsqueeze(1) + metaTE.unsqueeze(2)  # match dim and sum: [batch_size, num_hist+num_pred, num_vertex, D]

        # local
        # Wse, Wte = self.query_local_memory(TW_t)
        # Wse, Wte, query, att_score = self.query_local_memory(TW_t)
        # metaSE = torch.einsum('ne,bned->bnd', SE, Wse)
        # metaTE = torch.einsum('bte,bned->btnd', TE.to(device=self.device), Wte)
        # local_metaSTE = metaSE.unsqueeze(1) + metaTE      # match dim and sum: [batch_size, num_hist+num_pred, num_vertex, D]

        # metaSTE = torch.mul(torch.tanh(local_metaSTE), torch.sigmoid(global_metaSTE))
        # metaSTE = self.mem_fus(local_metaSTE, global_metaSTE)
        # metaSTE = self.mem_fus(local_metaSTE)
        # contrastive pos/neg
        #pos, neg = self.generate_NP_pairs(att_score)
        return global_metaSTE, query, pos, neg

        # SE = SE.unsqueeze(0).unsqueeze(0)
        # SE = self.FC_se(SE)
        # TE = TE.unsqueeze(dim=2).to(device=self.device)
        # TE = self.FC_te(TE)
        # return SE + TE


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class MemeTransformer(nn.Module):
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
    def __init__(self, SE, N, SE_dim, TE_dim, timestep_in, timestep_out, device, statt_layers=1, att_heads=8, att_dims=8, bn_decay=0.1,
                 mem_num:int=6, mem_dim:int=8, norm_num:int=3):
        super(MemeTransformer, self).__init__()
        L = statt_layers
        K = att_heads
        d = att_dims
        D = K * d
        self.num_his = timestep_in
        self.num_pred = timestep_out
        self.SE = SE
        self.STAttBlock_tw = nn.ModuleList([STAttBlock(K, d, 0, bn_decay, if_STE=False) for _ in range(L)])         # encoder twitter
        self.MetaSTEmbedding = MetaSTEmbedding(N, SE_dim, TE_dim, D, bn_decay, device)

        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, 0, bn_decay, if_STE=True) for _ in range(L)])          # encoder
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, 0, bn_decay, if_STE=True) for _ in range(L)])       # decoder
        # self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, mem_dim, bn_decay, if_STE=True) for _ in range(L)])    # decoder
        self.transformAttention = transformAttention(K, d, 0, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_tw = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)
        # self.FC_2 = FC(input_dims=[D+mem_dim, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)
        # memory use
        self.N = N
        self.D = D
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.st_memory = self.construct_st_memory()     # local memory: frame-wise
        self.seq_memory = self.construct_seq_memory()   # global memory: seq-wise
        # memory fusion
        self.mem_fus = gatedFusion(D+mem_dim, 0, bn_decay)
        # contrastive constraint
        self.mem_label = self.label_mem_sim(mem_num, norm_num).to(device)

    @staticmethod
    def label_mem_sim(mem_num:int, norm_num:int):
        assert norm_num <= mem_num

        norm = torch.ones(norm_num, norm_num)
        abno = torch.zeros(mem_num-norm_num, norm_num)
        label = torch.cat([norm, abno], dim=0)
        abno = torch.zeros(mem_num, mem_num-norm_num)
        label = torch.cat([label, abno], dim=1)
        return label

    def construct_seq_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.num_pred*self.N*self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim*self.N*self.num_pred), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_seq_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 4, 'Input to query Seq-Memory must be a 4D tensor'

        h_t_flat = h_t.reshape(h_t.shape[0], -1)    # (B, T*N*h)
        query = torch.mm(h_t_flat, self.seq_memory['Wa'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.seq_memory['memory'].t()), dim=1)         # alpha: (B, M)
        proto_t = torch.mm(att_score, self.seq_memory['memory'])    # (B, d)
        mem_t = torch.mm(proto_t, self.seq_memory['fc'])   # (B, T*N*d)
        _h_t = torch.cat([h_t, mem_t.reshape(h_t.shape[0], self.num_pred, self.N, self.mem_dim)], dim=-1)      # (B, T, N, h+d)

        # contrast: query -> anchor, most similar prototype -> positive, rest prototypes -> negative
        #pos, neg = self.generate_NP_pairs(att_score)
        # seq_mem_sim = torch.softmax(torch.mm(att_score.t(), att_score), dim=-1)  # exp(cos sim): (M, M)
        # seq_mem_sim = torch.softmax(torch.mm(self.seq_memory['memory'], self.seq_memory['memory'].t()), dim=-1)      # exp(cos sim): (M, M)
        return _h_t#, query, pos, neg

    def generate_NP_pairs(self, att_score:Tensor):      # for seq memory
        assert len(att_score.shape) == 2

        val, ind = att_score.sort(descending=True, dim=1)
        pos, neg = [], []
        for b in range(ind.shape[0]):
            b_neg = []
            for i in range(ind.shape[1]):
                if i == 0:      # most similar
                    pos.append(self.seq_memory['memory'][ind[b,i],:])
                else:
                    b_neg.append(self.seq_memory['memory'][ind[b,i],:])
            neg.append(torch.stack(b_neg, dim=1))
        pos = torch.stack(pos, dim=0)   # (B, d)
        neg = torch.stack(neg, dim=0)   # (B, d, M-1)
        return pos, neg

    def construct_st_memory(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.N, self.mem_num, self.mem_dim), requires_grad=True)     # (N, M, d)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.D, self.mem_dim), requires_grad=True)    # for project to query
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.mem_dim, self.mem_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'

        query = torch.einsum('bnh,hd->bnd', h_t, self.st_memory['Wa'])      # (B, N, d)
        att_score = torch.softmax(torch.einsum('bnd,nmd->bnm', query, self.st_memory['memory']), dim=-1)  # alpha: (B, N, M)
        proto_t = torch.einsum('bnm,nmd->bnd', att_score, self.st_memory['memory'])      # (B, N, d)
        mem_t = torch.matmul(proto_t, self.st_memory['fc'])     # (B, N, d)
        _h_t = torch.cat([h_t, mem_t], dim=-1)      # (B, N, h+d)

        #st_mem_sim = torch.softmax(torch.einsum('bnm,bnm->mm', att_score, att_score), dim=-1)       # exp(cos sim): (M, M)
        return _h_t#, st_mem_sim

    def forward(self, X, TE):
        # input
        X, TW = torch.unsqueeze(X[...,0], -1), torch.unsqueeze(X[...,-1], -1)    # channel 0: mob 1: twit
        X, TW = self.FC_1(X), self.FC_tw(TW)

        # tweet encoder
        for net in self.STAttBlock_tw:
            TW = net(TW, None)

        # STE
        STE, query, pos, neg = self.MetaSTEmbedding(TW, self.SE, TE)
        # STE = self.MetaSTEmbedding(TW, self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)

        # transAtt
        #X = torch.cat([X, TW], dim=-1)
        X = self.transformAttention(X, STE_his, STE_pred)
        #TW = self.transformAttention(TW, STE_his, STE_pred)

        # query seq memory
        # X_seq = self.query_seq_memory(X)
        # query ST memory
        # x = []
        # for t in range(self.num_pred):
        #     x.append(self.query_st_memory(X[:,t,:,:]))
        # X_st = torch.stack(x, dim=1)
        # fusion
        # X = self.mem_fus(X_seq, X_st)
        # X = torch.mul(X_st, torch.sigmoid(X_seq))

        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        X = torch.tanh(X)
        return torch.squeeze(X, -1), query, pos, neg
