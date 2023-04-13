#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
from trans_encoder import Encoder
#%%

class GeoNet(nn.Module):
    def __init__(self,kernel_size,num_filter,stride,num_feat = 14, d_time = 8):
        super(GeoNet,self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.process_coords = nn.Linear(2,16)
        self.embtime = nn.Embedding(1440,d_time)
        self.embtime.weight.data.copy_(torch.from_numpy(np.load('time.npy')))

        self.conv1 = nn.Conv1d(16,self.num_filter,self.kernel_size, stride = stride)
        self.conv2 = nn.Conv1d(num_feat,self.num_filter,self.kernel_size, stride = stride)
        self.conv3 = nn.Conv1d(d_time,self.num_filter,self.kernel_size, stride = stride)

    def forward(self,traj):
        lngs_lats = traj[:,:,[-2,-1]] 
        locs1 = torch.tanh(self.process_coords(lngs_lats))
        locs1 =locs1.permute(0,2,1)
        conv_locs1 = F.elu(self.conv1(locs1)).permute(0,2,1)
        
        features = traj[:,:,:-3]
        locs2 = features.permute(0,2,1)
        conv_locs2 = F.elu(self.conv2(locs2)).permute(0,2,1)
        
        time = traj[:,:,-3].long() 
        locs3 = torch.tanh(self.embtime(time))
        locs3 = locs3.permute(0,2,1)
        conv_locs3 = F.elu(self.conv3(locs3)).permute(0,2,1)
        return (conv_locs1,conv_locs2,conv_locs3)
        



class Transnet(nn.Module):
    def __init__(self,kernel_size,num_filter,stride,num_feat = 14, L_out=15,d_time = 8,d_model = 64,d_ff = 128,d_k = 64,d_v = 64,n_heads = 3,n_layers = 2,dropout = 0.5,embedding_size=256,num_classes = 10):
        super(Transnet, self).__init__()
        self.input_emb = GeoNet(kernel_size,num_filter,stride,num_feat, d_time)
        self.encoder1 = Encoder(d_model, d_ff,d_k,d_v,n_heads,n_layers)
        self.encoder2 = Encoder(d_model, d_ff,d_k,d_v,n_heads,n_layers)

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
        outputs, _ = self.encoder1(inputs_time,inputs_feat,inputs_feat)
        outputs, _ = self.encoder2(inputs_geo,outputs,outputs)

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
#%%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Alllittle') 
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--kernel_size', type=int, default=32)
    parser.add_argument('--num_filter', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--d_time', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lam1', type=float, default=1.0)
    parser.add_argument('--lam2', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args(args=[])
    train_data = torch.ones((4,512,14))

    args.num_feat = train_data.shape[2] - 3#)# L*seq_len*num_feat

    args.num_classes = 10
    args.d_model = args.num_filter
    args.L_out = int((train_data.shape[1] - (args.kernel_size -1) -1) / args.stride + 1)

    dev = 'cuda'.format(args.gpu) if (torch.cuda.is_available()) & (args.gpu>=0) else 'cpu'

    train_data=train_data.to(dev)
    met = Transnet(
        kernel_size = args.kernel_size,
        num_filter=args.num_filter,
        stride=args.stride,
        num_feat = args.num_feat,
        L_out = args.L_out,
        d_time=args.d_time,
        d_model = args.d_model,
        d_ff = args.d_ff,
        d_k = args.d_k,
        d_v= args.d_v,
        n_heads = args.n_heads,
        n_layers =args.n_layers,
        embedding_size=128,
        num_classes = args.num_classes).to(dev)
    
    print(met(train_data))


# %%
