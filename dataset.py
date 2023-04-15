import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

class FashionMNISTLoader:
    training_data = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    @staticmethod
    def get_random_inputs(input_count, data_loader = 'test'):
        if data_loader == 'test':
            data_loader = FashionMNISTLoader.test_dataloader
        elif data_loader == 'train':
            data_loader = FashionMNISTLoader.train_dataloader

        rand_arr = [random.randint(1, len(data_loader) - 1) for _ in range(input_count)]
        test_input_arr = [data_loader.dataset[i][0] for i in rand_arr]
        return test_input_arr
    
    @staticmethod
    def get_random_inputs_with_labels(input_count, data_loader = 'test'):
        if data_loader == 'test':
            data_loader = FashionMNISTLoader.test_dataloader
        elif data_loader == 'train':
            data_loader = FashionMNISTLoader.train_dataloader
            
        rand_arr = [random.randint(1, len(data_loader) - 1) for _ in range(input_count)]
        test_input_arr = [data_loader.dataset[i] for i in rand_arr]
        return test_input_arr
