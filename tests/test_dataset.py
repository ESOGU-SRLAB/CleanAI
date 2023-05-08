
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST    
from dataset import Dataset

# ----------------- #
# Dataset Class Tests

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


dataset = Dataset(training_data, test_data)

print(f'\n----------------------------------\n')
print(f'Get random input without label')
input = dataset.get_random_input()
print(input)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'Get random input with label')
input = dataset.get_random_input_with_label()
print(input)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'Get random inputs without labels')
input_arr = dataset.get_random_inputs(2)
print(input_arr)
print(f'\n----------------------------------\n')

print(f'\n----------------------------------\n')
print(f'Get random inputs with labels')
input_arr = dataset.get_random_inputs_with_labels(2)
print(input_arr)
print(f'\n----------------------------------\n')
