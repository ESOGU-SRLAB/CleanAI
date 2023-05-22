from model import load_model
from neural_network_profiler import NeuralNetworkProfiler
from dataset import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from coverage import Coverage
from model_architecture_utils import ModelArchitectureUtils
from math import inf

model = load_model("model.pth")

training_data = FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())


dataset = Dataset(training_data, test_data)
random_input = dataset.get_random_input()

print(model)

activation_info = NeuralNetworkProfiler.get_activation_info(model, random_input)
print(activation_info)


print(f"---------------------------------------\n")
print(f"Get threshold coverage for single layer (only last_layer): \n")
last_layer_after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
    activation_info, len(activation_info) - 1
)
print(Coverage.get_threshold_coverage_for_single_layer(last_layer_after_values, 0.1))

print(f"---------------------------------------\n")
print(f"Get threshold coverage for all layers: \n")
all_layers_after_values = ModelArchitectureUtils.get_after_values_for_all_layers(
    activation_info
)
print(Coverage.get_threshold_coverage_for_all_layers(all_layers_after_values, 0.1))

rnd_input_I = dataset.get_random_input()
rnd_input_II = dataset.get_random_input()

activation_info_for_tI = NeuralNetworkProfiler.get_activation_info(model, rnd_input_I)
activation_info_for_tII = NeuralNetworkProfiler.get_activation_info(model, rnd_input_II)

print(f"action info for tI: {activation_info_for_tI}")
print(f"action info for tII: {activation_info_for_tII}")
