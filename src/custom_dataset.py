import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

"""
Overview: This class is currently not used. The purpose of this class is 
to make the data set from the user ready for the use of the user and the 
program.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class CustomDataset(Dataset):
    def __init__(self, data_file, shape, dtype=torch.float32):
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
        """
        Reshapes the sample to the given shape and type.

        Args:
            sample: The sample to be reshaped.
            shape: The shape of the sample.
            dtype: The type of the sample.
        Returns:
            (Tensor): The reshaped sample.
        """
        if type(sample) != torch.Tensor:
            return sample

        sample = sample.numpy()
        sample = sample.reshape(shape)
        return torch.from_numpy(sample).type(dtype)

    def get_data(self, index):
        """
        Returns the entry in the given index from the data set.

        Args:
            index: The index of the entry to be returned.
        Returns:
            sample (Tensor): The sample in the given index.
            label (string): The label in the given index.
        """
        sample, label = self[index]
        sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)

        return sample, label

    def get_random_data(self):
        """
        Returns a random entry from the data set.

        Args:
            None
        Returns:
            sample (Tensor): The random sample.
            label (string): The random label.
        """
        random_index = random.randint(0, len(self) - 1)
        sample, label = self[random_index]
        sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)

        return sample, label

    def get_random_datas(self, n):
        """
        Returns random entries from the data set.

        Args:
            n (int): The number of random entries to be returned.
        Returns:
            datas (list): The list of random entries.
        """
        datas = []

        for i in range(n):
            sample, label = self.get_random_data()
            sample = CustomDataset.reshape_sample(sample, self.shape, self.dtype)
            datas.append((sample, label))

        return datas
