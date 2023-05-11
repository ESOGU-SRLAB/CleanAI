from neural_network_profiler import NeuralNetworkProfiler
from model import load_model
from dataset import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

training_data = FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())


dataset = Dataset(training_data, test_data)
model = load_model("model.pth")

print(f"---------------------------------------\n")
layer_names_arr = NeuralNetworkProfiler.get_layer_names(model)
print(f"Layer names: {layer_names_arr}\n")
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Run model with array: \n")
NeuralNetworkProfiler.run_model_with_array(dataset.get_random_input(), layer_names_arr)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get model architecture dict of input: \n")
model_architecture_dict_of_input = (
    NeuralNetworkProfiler.get_model_architecture_dict_of_input(
        dataset.get_random_input(), layer_names_arr
    )
)
print(model_architecture_dict_of_input)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get counter dict of model: \n")
counter_dict_of_model = NeuralNetworkProfiler.get_counter_dict_of_model(layer_names_arr)
print(counter_dict_of_model)
print(f"---------------------------------------\n")
