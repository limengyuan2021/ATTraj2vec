
variable_names =  ['speed','speed_add',
    'acc_lat','acc_lat_add',
    'acc_lng','acc_lng_add',
    'acc_jert',
    'bearing','bear_add','bear_rate',
    'time',
    'lat',
    'lng','Driver','ID']


#%%
import numpy as np
import _pickle as cPickle
import time
import progressbar#显示进度条
import numpy  as np
import os
from scipy import stats
import pandas as pd
import math
import argparse
from sklearn.preprocessing import MinMaxScaler

#%%

parser = argparse.ArgumentParser()

parser.add_argument('--L1', type=int, default=512)

args = parser.parse_args([])



data_path = 'data'
data_dir = 'data/traj.pkl'


#%%
def creat_subtracjectory(L1=args.L1):
    basicFeatures = pd.read_pickle(data_dir)

    bar = progressbar.ProgressBar()
    start = time.time()
    subtracjectory = {}
    for (Driver,ID), traj in bar(basicFeatures.groupby(['Driver','ID'])):#对于每一条轨迹
        
        matricesForTrajectory = []
        traj= traj.drop(columns=['Driver','ID']).values
        ranges = returnSegmentIndexes(L1, len(traj))        
        for p in ranges:
            if p[1] - p[0] == args.L1:#删掉小于L1的数据
                subtraj = traj[p[0]:p[1]]
                if sum(subtraj[:,0] ==0) < args.L1 /3:#至少有一半时间行驶
                    matricesForTrajectory.append(subtraj)#子轨迹 [seq_len,feat_len]
                     
        if len(matricesForTrajectory):
            subtracjectory['|'.join([str(int(Driver)),str(int(ID))])] = np.array(matricesForTrajectory)#[n,seq_len,feat_len]

      
    print("statistical features created",time.time()-start)

    save_path1 = '{}/data_temp.pkl'.format(data_path)
    save_path2 = '{}/data_temp.npy'.format(data_path)
    keys = [k.split("|") for k, v in subtracjectory.items() for _ in range(v.shape[0])]
    cPickle.dump(keys, open(save_path1, "wb"))
    save_data = np.vstack(list(subtracjectory.values()))

    np.save(save_path2, save_data, allow_pickle=False)

#划分子轨迹
def returnSegmentIndexes(L1, leng):
    ranges = []
    start = 0
    while True:        
        end = min(start+L1, leng-1)
        ranges.append([int(start), int(end)])
        start += L1/2
        if end == leng-1:
            break        
    return ranges


if __name__ == '__main__':
    creat_subtracjectory() 