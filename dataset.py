import random
import torch
import random
import torch
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, training_datas, testing_datas, batch_size = 64):
        self.train_data_loader = DataLoader(training_datas, batch_size)
        self.test_data_loader = DataLoader(testing_datas, batch_size)

    def get_data_loaders(self):
        return (self.train_data_loader, self.test_data_loader)
    
    def get_data_loader(self, data_loader_type = 'test'):
        if data_loader_type == 'test':
            return self.test_data_loader
        elif data_loader_type == 'train':
            return self.train_data_loader

    def get_random_input(self, data_loader_type = 'test'):
        data_loader = self.get_data_loader(data_loader_type)

        random_index = random.randint(1, len(data_loader) - 1)
        random_data = data_loader.dataset[random_index][0]
        return random_data
    
    def get_random_input_with_label(self, data_loader_type = 'test'):
        data_loader = data_loader = self.get_data_loader(data_loader_type)
            
        random_index = random.randint(1, len(data_loader) - 1)
        random_data = data_loader.dataset[random_index]
        return random_data

    def get_random_inputs(self, input_count, data_loader_type = 'test'):
        data_loader = self.get_data_loader(data_loader_type)

        rand_indexes_arr = [random.randint(1, len(data_loader) - 1) for _ in range(input_count)]
        random_datas = [data_loader.dataset[i][0] for i in rand_indexes_arr]
        return random_datas
    
    def get_random_inputs_with_labels(self, input_count, data_loader_type = 'test'):
        data_loader = self.get_data_loader(data_loader_type)
            
        rand_indexes_arr = [random.randint(1, len(data_loader) - 1) for _ in range(input_count)]
        random_datas = [data_loader.dataset[i] for i in rand_indexes_arr]
        return random_datas



