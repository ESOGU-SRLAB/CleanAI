import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, data_file, shape, dtype = torch.float32):
        self.data = pd.read_csv(data_file)
        self.shape = shape
        self.dtype = dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, 1:].values
        label = self.data.iloc[index, 0]

        sample = torch.from_numpy(np.array(sample))
        sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)
        label = torch.tensor(label)

        return sample, label
    
    @staticmethod
    def reshape_sample(sample, shape, dtype):
        if(type(sample) != torch.Tensor):
            return sample
        
        sample = sample.numpy()
        sample = sample.reshape(shape)
        return torch.from_numpy(sample).type(dtype)

    def get_data(self, index):
        sample, label = self[index]
        sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)

        return sample, label

    def get_random_data(self):
        random_index = random.randint(0, len(self) - 1)
        sample, label = self[random_index]
        sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)

        return sample, label
    
    def get_random_datas(self, n):
        datas = []

        for i in range(n):
            sample, label = self.get_random_data()
            sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)
            datas.append((sample, label))

        return datas


