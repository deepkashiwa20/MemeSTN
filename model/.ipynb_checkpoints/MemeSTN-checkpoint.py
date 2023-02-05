import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, torch.stack(output_hidden, dim=0)

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class AVWDCRNN_stepwise_decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN_stepwise_decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], node_embeddings)
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, torch.stack(output_hidden, dim=0)


class AGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units=64, num_layers=2, default_graph=True, embed_dim=8, cheb_k=2):
        super(AGCRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return torch.tanh(output)


class MemeSTN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units=64, num_layers=2, default_graph=True, embed_dim=8, cheb_k=2,
                 mem_num:int=10, mem_dim:int=8, tcov_in_dim=32, tcov_embed_dim=10, tcov_h_dim=1):
        super(MemeSTN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        # twitter branch
        self.encoder_tw = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)  # twi
        self.encoder = AVWDCRNN(num_nodes, input_dim+tcov_h_dim, rnn_units, cheb_k, embed_dim, num_layers)     # mob
        # tcov input
        self.tcov_in_dim = tcov_in_dim
        self.tcov_embed_dim = tcov_embed_dim
        self.tcov_h_dim = tcov_h_dim
        self.tcov_in = nn.Sequential(nn.Linear(tcov_in_dim, tcov_embed_dim, True),
                                     nn.Linear(tcov_embed_dim, num_nodes*tcov_h_dim, True))
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.global_memory = self.construct_global_memory()

        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def construct_global_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.num_node*self.hidden_dim, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['TE_FC1'] = nn.Parameter(torch.randn(self.mem_dim, self.tcov_in_dim*self.tcov_embed_dim), requires_grad=True)
        memory_dict['TE_FC2'] = nn.Parameter(torch.randn(self.mem_dim, self.tcov_embed_dim*self.num_node*self.tcov_h_dim), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_global_memory(self, TW_t:torch.Tensor):
        assert len(TW_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'
        B = TW_t.shape[0]

        TW_t_flat = TW_t.reshape(B, -1)
        query = torch.mm(TW_t_flat, self.global_memory['Wq'])     # (B, d)
        att_score = torch.softmax(torch.mm(query, self.global_memory['Memory'].t()), dim=1)         # alpha: (B, M)
        proto_t = torch.mm(att_score, self.global_memory['Memory'])     # (B, d)

        # memory-guided weight generation
        Wte1 = torch.matmul(proto_t, self.global_memory['TE_FC1'])
        Wte2 = torch.matmul(proto_t, self.global_memory['TE_FC2'])
        Wte1 = Wte1.reshape(B, self.tcov_in_dim, self.tcov_embed_dim)
        Wte2 = Wte2.reshape(B, self.tcov_embed_dim, self.num_node, self.tcov_h_dim)

        # generate pos/neg pair for feature loss
        _, ind = torch.topk(att_score, k=2, dim=1)
        pos = self.global_memory['Memory'][ind[:, 0]]
        neg = self.global_memory['Memory'][ind[:, 1]]
        return Wte1, Wte2, query, pos, neg

    # def forward(self, source, targets, teacher_forcing_ratio=0.5):
    def forward(self, source, TE):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # input
        X, TW = torch.unsqueeze(source[..., 0], -1), torch.unsqueeze(source[..., -1], -1)  # channel 0: mob 1: twi

        # twitter branch
        init_state = self.encoder_tw.init_hidden(TW.shape[0])
        output_tw, _ = self.encoder_tw(TW, init_state, self.node_embeddings)      # B, T, N, hidden
        output_tw = output_tw[:, -1, :, :]
        Wte1, Wte2, query, pos, neg = self.query_global_memory(output_tw)
        TE = torch.einsum('bti,bie->bte', TE, Wte1)     # layer 1: -> tcov embed
        # TE = torch.relu(TE)     # enhance inflows, decline outflows
        TE = torch.tanh(TE)
        TE = torch.einsum('bte,benh->btnh', TE, Wte2)   # layer 2: -> tcov hidden
        # TE = self.tcov_in(TE).reshape(TE.shape[0], TE.shape[1], self.num_node, self.tcov_h_dim)       # (B, a+b, N, tcov_h)
        X = torch.cat([X, TE[:,:-self.horizon,:,:]], dim=-1)

        # mobility branch
        init_state = self.encoder.init_hidden(X.shape[0])
        output, _ = self.encoder(X, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return torch.tanh(output), query, pos, neg