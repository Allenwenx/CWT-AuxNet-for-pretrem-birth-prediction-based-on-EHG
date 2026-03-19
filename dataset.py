import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, subset='train', data_dir='./data/samples'):
        super(MyDataset, self).__init__()
        txt_path = f'./data/{subset}.txt'
        self.samples = open(txt_path).readlines()
        self.data_dir = data_dir

    def __getitem__(self, index):
        line = self.samples[index].strip()
        sample_name, label = line.split()
        user_id = int(sample_name.split('_')[0])
        
        s2_path = os.path.join(self.data_dir, f'{sample_name}_S2.npy')
        s3_path = os.path.join(self.data_dir, f'{sample_name}_S3.npy')
        
        stat_feat_path = os.path.join(self.data_dir, f'{sample_name}_statfeat.npy')

        s2_tensor = np.load(s2_path).astype(np.float32)       # [H, W]
        s3_tensor = np.load(s3_path).astype(np.float32)       # [H, W]

        stat_feat = np.load(stat_feat_path).astype(np.float32)     


        s2_tensor = torch.from_numpy(s2_tensor).unsqueeze(0)  # [1, H, W]
        s3_tensor = torch.from_numpy(s3_tensor).unsqueeze(0)  # [1, H, W]

        stat_feat = torch.from_numpy(stat_feat)
        label = int(label)

        return {
            's2':s2_tensor,
            's3':s3_tensor,
            'feature': stat_feat,   
            'label': label,
            'user_id': user_id
        }

    def __len__(self):
        return len(self.samples)
