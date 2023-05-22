import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from neural_network_profiler import NeuralNetworkProfiler
from coverage import Coverage
from model_architecture_utils import ModelArchitectureUtils
from math import inf

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

resnet50 = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_resnet50", pretrained=True
)
utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_convnets_processing_utils"
)

resnet50.eval().to(device)

torch.save(resnet50.state_dict(), "nvidia_resnet50.pth")

uris = [
    "https://images.unsplash.com/photo-1560807707-8cc77767d783",
]

batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in uris]).to(device)

# print(batch[0])

with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256, 256), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    print(result)

activation_info = NeuralNetworkProfiler.get_activation_info(
    resnet50, batch[0].unsqueeze(0)
)
print(activation_info)
print(f"---------------------------------------\n")
last_layer_after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
    activation_info, len(activation_info) - 1
)
# print(f"Last layer: {last_layer_after_values}")
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get threshold coverage for single layer (only last_layer): \n")
(
    num_of_th_covered_neurons,
    total_neurons,
    threshold_coverage,
) = Coverage.get_threshold_coverage_for_single_layer(last_layer_after_values, 0.75)
print(
    f"Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
after_values_all_layers = ModelArchitectureUtils.get_after_values_for_all_layers(
    activation_info
)

print(f"Get threshold coverage for all layers: \n")
(
    num_of_th_covered_neurons,
    total_neurons,
    threshold_coverage,
) = Coverage.get_threshold_coverage_for_all_layers(after_values_all_layers, 0.1)
print(
    f"Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}"
)

print(f"---------------------------------------\n")
layer_9_after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
    activation_info, 9
)
# print(f"len(layer_9_after_values): {len(layer_9_after_values)}")
print(f"layer_9_after_values: {layer_9_after_values}")
(
    num_of_th_covered_neurons,
    total_neurons,
    threshold_coverage,
) = Coverage.get_threshold_coverage_for_single_layer(layer_9_after_values, 0.1)
print(
    f"Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}"
)

print(activation_info)
