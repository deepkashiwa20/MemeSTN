# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020
@author: wb
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 

class Transformer_EncoderBlock(nn.Module):
    def __init__(self, embed_size, pe_length, heads ,forward_expansion, gpu, dropout):
        super(Transformer_EncoderBlock, self).__init__()
        
        # Temporal embedding One hot
        self.pe_length = pe_length
#         self.one_hot = One_hot_encoder(embed_size, pe_length)          # temporal embedding by one-hot
        self.temporal_embedding = nn.Embedding(pe_length, embed_size)  # temporal embedding  by nn.Embedding
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.gpu = gpu
    def forward(self, value, key, query):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding by one-hot
        D_T = self.temporal_embedding(torch.arange(0, T).to(self.gpu))    # temporal embedding  by nn.Embedding
        D_T = D_T.expand(B, N, T, C)

        # temporal embedding + query。 (concatenated or add here is add)
        query = query + D_T  
#         print('query + D_T shape:',query.shape)

        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

### TTransformer_EncoderLayer
class TTransformer_EncoderLayer(nn.Module):
    def __init__(self, embed_size, pe_length, heads ,forward_expansion, gpu, dropout):
        super(TTransformer_EncoderLayer, self).__init__()
#         self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.Transformer_EncoderBlock = Transformer_EncoderBlock(embed_size, pe_length, heads ,forward_expansion, gpu, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.Transformer_EncoderBlock(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout(x1) 

        return x2

### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,embed_size,num_layers,pe_length,heads,forward_expansion,gpu,dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.gpu = gpu
        self.layers = nn.ModuleList([ TTransformer_EncoderLayer(embed_size, pe_length, heads ,forward_expansion, gpu, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):     
        # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)      
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out)
        return out     
    
### Transformer   
class T_Transformer_block(nn.Module):
    def __init__(self,embed_size,num_layers,pe_length,heads,forward_expansion, gpu,dropout):
        super(T_Transformer_block, self).__init__()
        self.encoder = Encoder(embed_size,num_layers,pe_length,heads,forward_expansion,gpu,dropout)
        self.gpu = gpu

    def forward(self, src): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src) 
        return enc_src # [B, N, T, C]


class Transformer(nn.Module):
    def __init__(
        self, in_channels, out_channels, embed_size, pe_length, num_layers, timestep_in, timestep_out, heads, forward_expansion, gpu, dropout):        
        super(Transformer, self).__init__()

        self.forward_expansion = forward_expansion
        # C --> expand  --> hidden dim (embed_size)
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        
        self.T_Transformer_block = T_Transformer_block(embed_size, num_layers, pe_length,heads, forward_expansion, gpu, dropout)

        # Reduce the temporal dimension。  timestep_in --> out_timestep_in
        self.conv2 = nn.Conv2d(timestep_in, timestep_out, 1)  
        # Reduce the C dimension，to 1。
        self.conv3 = nn.Conv2d(embed_size, out_channels, 1)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    
    def forward(self, x):
        # input x shape  [B, T, N, C]  C  = CHANNEL = 1
        # C: channel。  N:nodes。  T:time
        
        x = x.permute(0,3,2,1)   # [B, T, N, C] -> [B, C, N, T]
        input_Transformer = self.conv1(x)     #    x shape[B, C, N, T]   --->    input_Transformer shape： [B, H = embed_size = 64, N, T] 
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)    # [B, H, N, T] [B, N, T, H]
        output_Transformer = self.T_Transformer_block(input_Transformer)  # [B, N, T, H]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)   # [B, N, T, H] -> [B, T, N, H]
        
        out = self.relu(self.conv2(output_Transformer))    #   [B, T, N, H] ->  [B, T, N, C=1]         
        out = out.permute(0, 3, 2, 1)           # [B, T, N, C=1]  ->  [B, C=1, N, T]
        out = self.conv3(out)                   # 
        out = out.permute(0, 3, 2, 1)           # [B, C=1, N, T] -> [B,T,N,1]

        return out      #[B, TIMESTEP_OUT, N, C]   C = CHANNEL = 1
    
def print_params(model_name, model):
    param_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    print(f'{model_name}, {param_count} trainable parameters in total.')
    return  

def main():
    channel, his_len, seq_len=1, 12, 12
    GPU = '0'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Transformer(in_channels=channel, 
                        embed_size=64, 
                        pe_length=his_len, 
                        num_layers=1, 
                        timestep_in=his_len, 
                        timestep_out=seq_len, 
                        heads=8, 
                        forward_expansion=64, 
                        gpu=device, 
                        dropout=0).to(device)
    print_params('Transformer', model)
    summary(model, (his_len, 47, channel), device=device)
    
if __name__ == '__main__':
    main()
