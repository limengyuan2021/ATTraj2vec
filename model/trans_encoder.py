#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
#Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self, Q, K, V,d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        attn = F.Softmax(scores,dim=-1)
        return torch.matmul(attn, V) , attn

class Multi_Attention(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads):
        super(Multi_Attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.normatt = nn.LayerNorm(self.d_model)
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        

    def forward(self, input_Q, input_K, input_V):

        residual = input_V
        batch_size = input_V.shape[0]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  

        context, attn = Attention()(Q, K, V,self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) 
        
        output = self.fc(context) 
        return self.normatt(output + residual), attn


class PFF(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PFF, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.normpos = nn.LayerNorm(self.d_model)
        self.d_model = d_model
        
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return self.normpos(output + residual) 

class ONE_LAYER(nn.Module):
    def __init__(self,d_model, d_ff,d_k,d_v,n_heads):
        super(ONE_LAYER, self).__init__()
        self.enc_self_attn = Multi_Attention(d_model,d_k,d_v,n_heads)
        self.pos_ffn = PFF(d_model, d_ff)

    def forward(self, Q_inputs,K_inputs,V_inputs):

        enc_outputs, attn = self.enc_self_attn(Q_inputs,K_inputs,V_inputs)
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,d_model, d_ff,d_k,d_v,n_heads,n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ONE_LAYER(d_model, d_ff,d_k,d_v,n_heads) for _ in range(n_layers)])

    def forward(self, Q_inputs,K_inputs,V_inputs):

        enc_self_attns = []
        for i,layer in enumerate(self.layers):
            if i == 0:
                enc_inputs, enc_self_attn = layer(Q_inputs,K_inputs,V_inputs)
            else:
                enc_inputs, enc_self_attn = layer(enc_inputs,enc_inputs,enc_inputs)
            enc_self_attns.append(enc_self_attn)
        return enc_inputs, enc_self_attns
