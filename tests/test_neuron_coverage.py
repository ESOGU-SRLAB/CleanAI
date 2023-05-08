from model_architecture_utils import ModelArchitectureUtils
from neural_network_profiler import NeuralNetworkProfiler
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from neuron_coverage import NeuronCoverage
from model import load_model
from dataset import Dataset

training_data = FashionMNIST(
root='data',
train=True,
download=True,
transform=ToTensor()
)

test_data = FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)


model = load_model('model.pth')
model_layers_arr = NeuralNetworkProfiler.get_layer_names(model)

dataset = Dataset(training_data, test_data)
random_input = dataset.get_random_input()

model_architecture_dict = NeuralNetworkProfiler.get_model_architecture_dict_of_input(random_input, model_layers_arr)
after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(model_architecture_dict, 0)
after_values_all_layers = ModelArchitectureUtils.get_after_values_for_all_layers(model_architecture_dict)

print(f'---------------------------------------\n')
print(f'Get neuron coverage for single layer: \n')
num_of_covered_neurons, total_neurons, neuron_coverage = NeuronCoverage.get_neuron_coverage_for_single_layer(after_values)
print(f'Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}')
print(f'---------------------------------------\n')

print(f'---------------------------------------\n')
print(f'Get neuron coverage for all layers: \n')
num_of_covered_neurons, total_neurons, neuron_coverage = NeuronCoverage.get_neuron_coverage_for_all_layers(after_values_all_layers)
print(f'Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}')
print(f'---------------------------------------\n')

print(f'---------------------------------------\n')
print(f'Get threshold coverage for single layer: \n')
num_of_th_covered_neurons, total_neurons, threshold_coverage = NeuronCoverage.get_threshold_coverage_for_single_layer(after_values)
print(f'Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}')
print(f'---------------------------------------\n')

print(f'---------------------------------------\n')
print(f'Get threshold coverage for all layers: \n')
num_of_th_covered_neurons, total_neurons, threshold_coverage = NeuronCoverage.get_threshold_coverage_for_all_layers(after_values_all_layers)
print(f'Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}')
print(f'---------------------------------------\n')