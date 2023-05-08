from model_architecture_utils import ModelArchitectureUtils
from neural_network_profiler import NeuralNetworkProfiler
from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from model import load_model

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

dataset = Dataset(train_dataloader, test_dataloader)
random_input = dataset.get_random_input()

model = load_model('model.pth')
model_layers_arr = NeuralNetworkProfiler.get_layer_names(model)

model_architecture_dict = NeuralNetworkProfiler.get_model_architecture_dict_of_input(random_input, model_layers_arr)

print(f'\n----------------------------------\n')
print(f'Before values for specific layer: ')
before_values_for_specific_layer = ModelArchitectureUtils.get_before_values_for_specific_layer(model_architecture_dict, 0)
print(before_values_for_specific_layer)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'Before values for all layers: ')
before_values_for_all_layers = ModelArchitectureUtils.get_before_values_for_all_layers(model_architecture_dict)
print(before_values_for_all_layers)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'After values for specific layer: ')
after_values_for_specific_layer = ModelArchitectureUtils.get_after_values_for_specific_layer(model_architecture_dict, 0)
print(after_values_for_specific_layer)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'After values for all layers: ')
after_values_for_all_layers = ModelArchitectureUtils.get_after_values_for_all_layers(model_architecture_dict)
print(after_values_for_all_layers)
print(f'\n----------------------------------\n')