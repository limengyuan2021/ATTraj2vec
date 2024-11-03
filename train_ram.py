#%%
import _pickle as cPickle
import numpy as np
import random
import argparse
import time
import torch
from torch.utils.data import Dataset
import models.model as model
import torch
import datetime
from sklearn.cluster import KMeans,FeatureAgglomeration,AffinityPropagation,DBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_score,calinski_harabasz_score,adjusted_rand_score,adjusted_mutual_info_score
import torch.nn as nn
from train_dataram import get_data





#%%
#这里预定义了一些超参数
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='data_temp', help='the name of dataset') 
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=64)
parser.add_argument('--num_filter', type=int, default=64)
parser.add_argument('--stride', type=int, default=32)
parser.add_argument('--d_time', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--d_k', type=int, default=64)
parser.add_argument('--d_v', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--epoch', type=int, default=4000)
parser.add_argument('--lam1', type=float, default=1.0)
parser.add_argument('--lam2', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--init', type=int, default=1)
parser.add_argument('--de', type=str, default='xxxxxxx') 
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--type', type=str, default='v')
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--sqe', type=int, default=123)

#args = parser.parse_args(args=[])
args = parser.parse_args()

args.sqe = [int(digit)-1 for digit in str(args.sqe)]
print(args.sqe)


dic = ['speed','speed_add',
    'acc_lat','acc_lat_add',
    'acc_lng','acc_lng_add',
    'acc_jert',
    'bearing','bear_add','bear_rate',
    'time',
    'lat',
    'lng']

#lls = [i for i in range(len(dic)) if ((args.de not in dic[i]) )]

lls = [i for i in range(len(dic)) if (args.de not in dic[i]) ]








    




#这里定义了数据集
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
    #Train: For each sample creates randomly a positive or a negative pair
    #Test: Creates fixed pairs for testing
    def __init__(self, data,is_train = True):

        self.drive_dataset = data.data
        self.is_train = is_train
        self.labels = data.labels#标签
        self.labels_set = set(self.labels)#标签有哪些
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}#每个标签对应的位置
        if not self.is_train:#测试集还会多一个test_pairs
            # generate fixed pairs for testing
            random_state = np.random.RandomState(1)#固定的种子

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i]]),1] \
                              for i in range(0, len(self.drive_dataset), 2)]#每隔两个选一个正的

            negative_pairs = [[i,random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.labels[i]])))]),0] \
                              for i in range(1, len(self.drive_dataset), 2)]#每个两个选一个负的
                  
            self.test_pairs = positive_pairs + negative_pairs
            random_state.shuffle(self.test_pairs)

    def __getitem__(self, index):#
        if self.is_train:
            target = np.random.randint(0, 2)
            img1, label1 = self.drive_dataset[index], self.labels[index]#选出数据和标签
            if target == 1:
                siamese_index = index
                while siamese_index == index:#不能是自己
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



#这里定义了训练函数
def train(model, train_loader, optimizer, loss_fn, loss_fn2):
    epoch_loss = 0
    total_len = 0
    loss1s = []
    loss2s = []
    corrects1 = 0#这个是分类的错误率
    corrects2 = 0#这个是二分类的错误率
    model.train() #model.train()代表了训练模式    
    for img1, img2, label1,label2,target in train_loader: 
        target = target.to(device).float()
        img1 = img1.to(device)
        img2 = img2.to(device)
        #labels = torch.cat([label1,label2],0).to(device).long()
        label1 = label1.long().to(device)
        
        out1_emb = model(img1)#两个数据的embedding
        out2_emb = model(img2)#两个数据的embedding

        
        out1 = model.Classfier(out1_emb)

        ########class_loss        
        loss1 = loss_fn(out1,label1)

        #####Siameseset_loss
        score = model.Similarity_Score(out1_emb,out2_emb)
        loss2 = loss_fn2(score,target)
        if args.lam2>=0:
            loss = args.lam1 *loss1 + args.lam2 * loss2
        else:
            loss = args.lam1 *loss1
        loss1s.append(loss1)
        loss2s.append(loss2)

        optimizer.zero_grad() #加这步防止梯度叠加
        loss.backward() #反向传播
        optimizer.step() #梯度下降
        
        epoch_loss += loss.item() * len(target)
        total_len += len(target)


        ##计算准确率
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
            #labels = torch.cat([label1,label2],0).to(device).long()
            label1 = label1.long().to(device)

            
            out1_emb = model(img1)
            out2_emb = model(img2)

            score = model.Similarity_Score(out1_emb,out2_emb)
            out1 = model.Classfier(out1_emb)

            ########class_loss            
            loss1 = loss_fn(out1,label1)

            #####Siameseset_loss
            loss2 = loss_fn2(score,target)

            if args.lam2>=0:
                loss = args.lam1 *loss1 + args.lam2 * loss2
            else:
                loss = args.lam1 *loss1
            loss1s.append(loss1)
            loss2s.append(loss2)
            
            epoch_loss += loss.item() * len(target)
            total_len += len(target)


            ##计算准确率
            _,pred = torch.max(out1.data,1) 
            corrects1 += (pred == label1).sum().item()

            pred2 = (score > 0.5) * 1
            corrects2 += (pred2 == target).sum().item()
    model.train()
        
    return epoch_loss / total_len, corrects1/total_len  ,corrects2/total_len


def cluster(care,labels):
    with torch.no_grad():
        net.eval()
        ll = [net.Similarity_Score(care,care[i].expand(care.shape[0],care.shape[1]))   for i in range(care.shape[0])]
    ll = torch.stack(ll)

    distance_matrix = 1 - (ll.cpu().numpy() + ll.cpu().numpy().T)/2
    agglo = AgglomerativeClustering(n_clusters = None,distance_threshold = 0.5,linkage = 'complete',affinity = 'precomputed').fit(distance_matrix)
    print(set(agglo.labels_))
    print(adjusted_mutual_info_score(labels, agglo.labels_))#互信息




if __name__ == '__main__':
    
    st  =time.time()

    print('seed',args.seed)
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_tripId, sum_classes = get_data(k=args.k,seed = args.seed)
    train_data = train_data[:,:,lls]
    dev_data = dev_data[:,:,lls]
    test_data = test_data[:,:,lls]

    args.num_feat = train_data.shape[2] - 3#)# L*seq_len*num_feat
    #总的特征
    args.num_classes = sum_classes
    args.d_model = args.num_filter
    args.L_out = int((train_data.shape[1] - (args.kernel_size -1) -1) / args.stride + 1)
    args.now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%f')
    print('-------------------------------------------------------\n',args,'\n------------\n')


    trainset = TripDataset(train_data, train_labels)
    devset = TripDataset(dev_data, dev_labels )
    testset = TripDataset(test_data, test_labels)

    train_pairs = Siameseset(trainset,is_train=True)#训练集
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
        args = args,
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
    step = 0
    for epoch in range(args.epoch):
        train_loss,class_acc,sim_acc = train(net, train_loader, optimizer, loss_fn,loss_fn2)    

        print("epoch:",epoch,"train_loss:",train_loss,'class_acc',class_acc,'sim_acc',sim_acc)

        dev_loss,class_acc,sim_acc = evaluate(net, dev_loader, loss_fn,loss_fn2)    
        print("epoch:",epoch,"dev_loss:",dev_loss,'class_acc',class_acc,'sim_acc',sim_acc)
        if (class_acc + sim_acc) > best_dev_acc:
            best_dev_acc = (class_acc + sim_acc)
            torch.save(net.state_dict(), 'save_model/best_model{}{}{}.pt'.format(args.data_name,args.now,args.seed))
            print('saved best')
            step = 0
        else:
            step +=1
            if step > 500:
                print(f'Early stopping after {epoch+1} epochs.')
                break
    torch.save(net.state_dict(), 'save_model/epoch{}_model{}{}{}.pt'.format(args.epoch,args.data_name,args.now,args.seed))  
    net.load_state_dict(torch.load('save_model/best_model{}{}{}.pt'.format(args.data_name,args.now,args.seed)))
    print('save_model/best_model{}{}{}.pt'.format(args.data_name,args.now,args.seed))
    dev_loss,class_acc,sim_acc = evaluate(net, test_loader, loss_fn,loss_fn2) 
    print("test_loss:",dev_loss,'class_acc',class_acc,'sim_acc',sim_acc)
    print(' {:.1f} seconds!'.format(time.time()-st))
    
    # tests.append(class_acc)
    if args.lam1 == 0:
        with torch.no_grad():
            net.eval()
            scores = []
            for i in range(sum_classes):
                data = train_data[train_labels==i]
                emb = net(torch.tensor(data).to(device))
                emb = torch.mean(emb,0).reshape(1,-1)
                score = net.Similarity_Score(net(torch.tensor(test_data).to(device)),emb.expand(test_data.shape[0],emb.shape[1]))
                scores.append(score)
        _,pred =torch.max(torch.stack(scores),0)
        pred=pred.cpu().numpy()
        temp = np.mean(pred == test_labels)
        print(temp)
        



# %%
