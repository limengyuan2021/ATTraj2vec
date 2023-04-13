#%%
import _pickle as cPickle
import numpy as np
import random
import argparse
import time
import torch
from torch.utils.data import Dataset

import torch
import datetime
import model
import torch.nn as nn
from train_dataram import get_data


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='data10_134', help='the name of dataset') 
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=64)
parser.add_argument('--num_filter', type=int, default=64)
parser.add_argument('--stride', type=int, default=32)
parser.add_argument('--d_time', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--d_k', type=int, default=64)
parser.add_argument('--d_v', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=3)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--epoch', type=int, default=4000)
parser.add_argument('--lam1', type=float, default=1.0)
parser.add_argument('--lam2', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--init', type=int, default=1)
parser.add_argument('--de', type=str, default='xxxxxxx', help='the name of dataset') 
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--type', type=str, default='all')

args = parser.parse_args()
print(args.de)




class  TripDataset(Dataset):
    def __init__(self, data,labels,tripId=None):
        super(TripDataset, self).__init__()
        self.data = torch.Tensor(data)
        self.labels = labels
        self.tripId = tripId
    def __len__(self):
        return len(self.labels)
    def  __getitem__(self, i):
        if self.tripId is None:    
            return self.data[i],self.labels[i]
        else:
            return self.data[i],self.labels[i],self.tripId[i]
class Siameseset(Dataset):

    def __init__(self, data,is_train = True):

        self.drive_dataset = data.data
        self.is_train = is_train
        self.labels = data.labels
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
        if not self.is_train:

            random_state = np.random.RandomState(1)

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i]]),1] \
                              for i in range(0, len(self.drive_dataset), 2)]

            negative_pairs = [[i,random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.labels[i]])))]),0] \
                              for i in range(1, len(self.drive_dataset), 2)]
                  
            self.test_pairs = positive_pairs + negative_pairs
            random_state.shuffle(self.test_pairs)

    def __getitem__(self, index):#
        if self.is_train:
            target = np.random.randint(0, 2)
            img1, label1 = self.drive_dataset[index], self.labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
                    label2 = label1
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
                label2 = siamese_label
            img2 = self.drive_dataset[siamese_index]
            
        else:
            img1 = self.drive_dataset[self.test_pairs[index][0]]
            img2 = self.drive_dataset[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            label1 = self.labels[self.test_pairs[index][0]]
            label2 = self.labels[self.test_pairs[index][1]]

        return img1, img2, label1,label2,target

    def __len__(self):
        return len(self.labels)



#
def train(model, train_loader, optimizer, loss_fn, loss_fn2):
    epoch_loss = 0
    total_len = 0
    loss1s = []
    loss2s = []
    corrects1 = 0#
    corrects2 = 0#
    model.train() #model.train()    
    for img1, img2, label1,label2,target in train_loader: 
        target = target.to(device).float()
        img1 = img1.to(device)
        img2 = img2.to(device)

        label1 = label1.long().to(device)
        
        out1_emb = model(img1)#
        out2_emb = model(img2)#

        
        out1 = model.Classfier(out1_emb)

      
        loss1 = loss_fn(out1,label1)


        score = model.Similarity_Score(out1_emb,out2_emb)
        loss2 = loss_fn2(score,target)

        loss = args.lam1 *loss1 + args.lam2 * loss2
        loss1s.append(loss1)
        loss2s.append(loss2)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        epoch_loss += loss.item() * len(target)
        total_len += len(target)


        
        _,pred = torch.max(out1.data,1) 
        corrects1 += (pred == label1).sum().item()

        pred2 = (score > 0.5) * 1
        corrects2 += (pred2 == target).sum().item()
        
    return epoch_loss / total_len, corrects1/total_len ,corrects2/total_len

def evaluate(model, dev_loader, loss_fn, loss_fn2):    
    epoch_loss = 0
    total_len = 0
    loss1s = []
    loss2s = []
    corrects1 = 0
    corrects2 = 0
    model.eval()  
    with torch.no_grad():
        for img1, img2, label1,label2,target in dev_loader: 
            target = target.to(device).float()
            img1 = img1.to(device)
            img2 = img2.to(device)
            label1 = label1.long().to(device)

            
            out1_emb = model(img1)
            out2_emb = model(img2)

            score = model.Similarity_Score(out1_emb,out2_emb)
            out1 = model.Classfier(out1_emb)

            loss1 = loss_fn(out1,label1)

            loss2 = loss_fn2(score,target)

            loss = args.lam1 * loss1 + args.lam2 * loss2
            loss1s.append(loss1)
            loss2s.append(loss2)
            
            epoch_loss += loss.item() * len(target)
            total_len += len(target)


            _,pred = torch.max(out1.data,1) 
            corrects1 += (pred == label1).sum().item()

            pred2 = (score > 0.5) * 1
            corrects2 += (pred2 == target).sum().item()
    model.train()
        
    return epoch_loss / total_len, corrects1/total_len  ,corrects2/total_len


if __name__ == '__main__':
    tests =[]        
    st  =time.time()
    args.seed = np.random.choice(100000,1).item()
    print('seed',args.seed)
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_tripId, sum_classes =cPickle.load(open('/root/autodl-tmp/{}.pkl'.format(args.data_name), 'rb'))

    args.num_feat = train_data.shape[2] - 3
    args.num_classes = sum_classes
    args.d_model = args.num_filter
    args.L_out = int((train_data.shape[1] - (args.kernel_size -1) -1) / args.stride + 1)
    args.now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    print('-------------------------------------------------------\n',args,'\n------------\n')
#%%

    trainset = TripDataset(train_data, train_labels)
    devset = TripDataset(dev_data, dev_labels )
    testset = TripDataset(test_data, test_labels)

    train_pairs = Siameseset(trainset,is_train=True)#
    dev_pairs = Siameseset(devset,is_train=False)
    test_pairs = Siameseset(testset,is_train=False)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = args.num_workers
        )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = args.num_workers
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = args.num_workers
        )

    device = 'cuda:{}'.format(args.gpu) if (torch.cuda.is_available() & args.gpu>=0) else 'cpu'
    print('device',device)
    net = model.Transnet(
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
        num_classes = args.num_classes).to(device)
    if args.init:
        def weigth_init(m):
            if isinstance(m, nn.Conv1d):
                print(m)
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data,0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print(m)
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_() 
        net.apply(weigth_init)   

    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,eps=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn2 = torch.nn.BCELoss()
    best_dev_acc =0
    for epoch in range(args.epoch):
        train_loss,class_acc,sim_acc = train(net, train_loader, optimizer, loss_fn,loss_fn2)    

        print("epoch:",epoch,"train_loss:",train_loss,'class_acc',class_acc,'sim_acc',sim_acc)

        dev_loss,class_acc,sim_acc = evaluate(net, dev_loader, loss_fn,loss_fn2)    
        print("epoch:",epoch,"dev_loss:",dev_loss,'class_acc',class_acc,'sim_acc',sim_acc)
        if (class_acc + sim_acc) > best_dev_acc:
            best_dev_acc = (class_acc + sim_acc)
            torch.save(net.state_dict(), '/root/autodl-tmp/traj/save_model/best_model{}{}{}{}.pt'.format(args.data_name,args.now,args.seed,jj))
            print('saved best')
    torch.save(net.state_dict(), '/root/autodl-tmp/traj/save_model/epoch{}_model{}{}{}{}.pt'.format(args.epoch,args.data_name,args.now,args.seed,jj))  
    net.load_state_dict(torch.load('/root/autodl-tmp/traj/save_model/best_model{}{}{}{}.pt'.format(args.data_name,args.now,args.seed,jj)))
    print('/root/autodl-tmp/traj/save_model/best_model{}{}{}{}.pt'.format(args.data_name,args.now,args.seed,jj))
    dev_loss,class_acc,sim_acc = evaluate(net, test_loader, loss_fn,loss_fn2) 
    print("test_loss:",dev_loss,'class_acc',class_acc,'sim_acc',sim_acc)
    print(' {:.1f} seconds!'.format(time.time()-st))



# %%
