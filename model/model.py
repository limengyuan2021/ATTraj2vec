#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
from .transformer import Encoder
#%%

time_dir = 'models/time.npy'

class GeoNet(nn.Module):
    def __init__(self,kernel_size,num_filter,stride,num_feat = 14, d_time = 8):
        super(GeoNet,self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.process_coords = nn.Linear(2,16)#将2维的地理转化为16维
        self.embtime = nn.Embedding(1440,d_time)
        self.embtime.weight.data.copy_(torch.from_numpy(np.load(time_dir)))

        self.conv1 = nn.Conv1d(16,self.num_filter,self.kernel_size, stride = stride)
        self.conv2 = nn.Conv1d(num_feat,self.num_filter,self.kernel_size, stride = stride)
        self.conv3 = nn.Conv1d(d_time,self.num_filter,self.kernel_size, stride = stride)

    def forward(self,traj):# traj:batch_size*seq_len*17
        # 地理卷积
        lngs_lats = traj[:,:,[-2,-1]] #batch_size*seq_len*2
        locs1 = torch.tanh(self.process_coords(lngs_lats))# batch_size*seq_len*16
        locs1 =locs1.permute(0,2,1)# batch_size*16*seq_len
        conv_locs1 = F.elu(self.conv1(locs1)).permute(0,2,1)# L*seq_len'*num_filter
        
        # 特征卷积
        features = traj[:,:,:-3]# batch_size*seq_len*14
        locs2 = features.permute(0,2,1)# batch_size*14*seq_len
        conv_locs2 = F.elu(self.conv2(locs2)).permute(0,2,1)# L*seq_len'*num_filter
        
        # 时间卷积
        time = traj[:,:,-3].long() # batch_size*seq_len
        locs3 = torch.tanh(self.embtime(time))# batch_size*seq_len*d_time
        locs3 = locs3.permute(0,2,1)# batch_size*d_time*seq_len
        conv_locs3 = F.elu(self.conv3(locs3)).permute(0,2,1)# L*seq_len'*num_filter
        return (conv_locs1,conv_locs2,conv_locs3)#地理、特征、时间
        ## L*seq_len'*num_filter





class Transnet(nn.Module):
    def __init__(self,args,kernel_size,num_filter,stride,num_feat = 14, L_out=15,d_time = 8,d_model = 64,d_ff = 128,d_k = 64,d_v = 64,n_heads = 3,n_layers = 2,dropout = 0.5,embedding_size=256,num_classes = 10):
        super(Transnet, self).__init__()
        self.args = args
        self.input_emb = GeoNet(kernel_size,num_filter,stride,num_feat, d_time)# L*seq_len'*num_filter
        self.encoder1 = Encoder(d_model, d_ff,d_k,d_v,n_heads,n_layers)# L*seq_len'*d_v
        self.encoder2 = Encoder(d_model, d_ff,d_k,d_v,n_heads,n_layers)# L*seq_len'*d_v

        self.fc = nn.Linear(d_v*L_out,embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.Similarity_Score_fc = nn.Sequential(
            nn.Linear(embedding_size*2,embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_size,1)
        )

        self.Classfier_fc = nn.Sequential(
            nn.Linear(embedding_size,embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_size,num_classes)
        )

    def forward(self, inputs):
        inputs_geo,inputs_feat,inputs_time = self.input_emb(inputs)
        
        inputs_geo = self.dropout(inputs_geo)
        inputs_feat = self.dropout(inputs_feat)
        inputs_time = self.dropout(inputs_time)
        # Q_inputs,K_inputs,V_inputs


        '''
        0:feat
        1:time
        2:geo
        '''
        features = [inputs_feat,inputs_time,inputs_geo]
        V = features[self.args.sqe[0]]
        Q1 = features[self.args.sqe[1]]
        Q2 = features[self.args.sqe[2]]

        outputs, _ = self.encoder1(Q1,V,V)
        outputs, _ = self.encoder2(Q2,outputs,outputs)

        outputs_emb = F.relu(self.fc(outputs.reshape(outputs.shape[0],-1)))

        return outputs_emb
    
    def Classfier(self,out):
        x = self.dropout(out)
        x = self.Classfier_fc(x)
        return x
    
    def Similarity_Score(self,out1,out2):
        x = self.dropout(torch.cat([out1,out2],1)) 
        x = self.Similarity_Score_fc(x)
        x = torch.sigmoid(x).view(-1)
        return x