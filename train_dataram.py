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
from sklearn.cluster import KMeans,FeatureAgglomeration,AffinityPropagation,DBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_score,calinski_harabasz_score,adjusted_rand_score,adjusted_mutual_info_score


def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
def returnTrainDevTestData(keys,matrices):
    st  =time.time()
    # to split data to train, dev, and test; default: 75% train, 10% dev, and 15% test



    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    test_tripId = []
    
    curTraj = ''
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:#如果这个司机在集合中
            dr = driverIds[d]#dr是这个司机的位置
        else: 
            dr = len(driverIds)#dr是司机的位置
            driverIds[d] = dr#d对应第几个司机
        m = matrices[idx]#matrices的第idx个数据
        if t != curTraj:#轨迹变了就从新生成随机数
            curTraj = t
            r = random.random()
        if r < 0.75: 
            train_data.append(m)
            train_labels.append(dr)
        elif r < 0.85: 
            dev_data.append(m)
            dev_labels.append(dr)
        else: 
            test_data.append(m)
            test_labels.append(dr)      
            test_tripId.append(t)

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data   = np.asarray(dev_data, dtype="float32")
    dev_labels = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)
    print('Train, Test datasets are loaded in {:.1f} seconds!'.format(time.time()-st))
    print('There are {} samples in train, {} in dev, and {} in test set!'.format(len(train_data), len(dev_data), len(test_data)))
    print('num_drivers', len(driverIds) )  
    
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_tripId, len(driverIds) 


# %%
def get_data(k=10,seed = 1):


    matrices1 = load_data('data/data_temp.npy')
    keys1 = cPickle.load(open('data/data_temp.pkl', 'rb'))
    keys1 = np.array(keys1).astype('int')

    np.random.seed(seed)
    random.seed(seed)
    idx = np.random.choice(np.unique(keys1[:,0]),k, replace=False)
    keys = np.concatenate([keys1[np.isin(keys1[:,0], idx)]],0 )
    matrices = np.concatenate([matrices1[np.isin(keys1[:,0], idx)]],0 )


    data_250 = returnTrainDevTestData(keys,matrices)
    return data_250
