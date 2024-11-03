#计算更新后的V

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
#Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch


#计算更新后的V
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.d_model = d_model
        self.normatt = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_V, input_V.shape[0]
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V,self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.normatt(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.normpos = nn.LayerNorm(self.d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.normpos(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,d_model, d_ff,d_k,d_v,n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, Q_inputs,K_inputs,V_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs, attn = self.enc_self_attn(Q_inputs,K_inputs,V_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,d_model, d_ff,d_k,d_v,n_heads,n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff,d_k,d_v,n_heads) for _ in range(n_layers)])
        #d_model, #
        # d_ff  FeedForward dimension
    def forward(self, Q_inputs,K_inputs,V_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_self_attns = []
        for i,layer in enumerate(self.layers):
            if i == 0:
                enc_inputs, enc_self_attn = layer(Q_inputs,K_inputs,V_inputs)
            else:
                enc_inputs, enc_self_attn = layer(enc_inputs,enc_inputs,enc_inputs)
            enc_self_attns.append(enc_self_attn)
        return enc_inputs, enc_self_attns