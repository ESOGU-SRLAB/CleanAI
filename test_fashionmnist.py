import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

from driver import Driver
from custom_dataset import CustomDataset
from print_utils import PDFWriter
from definitions import *

# Device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Analiz sonuçlarının kaydedileceği PDF dosyasının ismi
pdf_file_name = "Analysis_FashionMNIST.pdf"
pdf_writer = PDFWriter(pdf_file_name)
pdf_writer.add_text("CleanAI Analysis Report", font_size=20, is_bold=True)

# FashionMNIST veri setinin CSV dosyasının yolu
data_file = "fashion-mnist_test.csv"

# CustomDataset sınıfını kullanarak veri setini oluşturun
custom_dataset = CustomDataset(data_file, (1, 28, 28), torch.float32)

# FashionMNIST modelini yükleme
model = torch.load("./model_fashion_1.pth")
input_idx = 25
sample, label = custom_dataset.get_data(input_idx)

# Driver sınıfını kullanarak modeli ve veri setini kullanarak bir sürücü oluşturun
driver = Driver(model, custom_dataset)

# Modelin bilgilerini alın
model_info = driver.get_model_informations()

# Modelin bilgilerini PDF'e kaydedin
data = [
    ["", "Model name", "Total params", "Number of layers"],
    [
        "Informations",
        model_info["name"],
        str(model_info["total_params"]),
        str(model_info["num_layers"]),
    ],
]

pdf_writer.add_text("Model Informations", font_size=16, is_bold=True)
pdf_writer.add_text(
    "The table below shows general information about the '"
    + model_info["name"]
    + "' model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)
pdf_writer.add_table(data)
pdf_writer.add_space(20)

# Modelin katmanlarının kapsamını hesaplayın (Neuron coverage layer by layer)


# Modelin katmanlarının kapsamını PDF'e kaydedin (neuron coverage)
data = [
    [
        "Layer index",
        "Activation function",
        "Number of covered neurons",
        "Number of total neurons",
        "Coverage value",
        "Mean of layer",
    ]
]
coverage_values_of_layers = driver.get_coverage_of_layers(input_idx)
mean_of_layer = coverage_values_of_layers[0][2]
pdf_writer.add_text(
    "Coverage Values of Layers Mean Value = " + str(mean_of_layer),
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows coverage values about the '"
    + model_info["name"]
    + "' model's all layers. NOTE: The coverage value of a layer is the ratio of the number of covered neurons to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the "
    + str(input_idx)
    + "th input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

for idx, (
    layer_index,
    activation_fn,
    mean_of_layer,
    num_of_covered_neurons,
    total_neurons,
    coverage,
) in enumerate(coverage_values_of_layers):
    data.append(
        [
            "Layer " + str(layer_index),
            str(activation_fn),
            str(num_of_covered_neurons),
            str(total_neurons),
            f"{coverage * 100:.2f}%",
            str(mean_of_layer),
        ]
    )

num_of_covered_neurons, total_neurons, coverage = driver.get_coverage_of_model(
    input_idx
)
data.append(
    [
        "All model",
        "-",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)
pdf_writer.add_table(data)
pdf_writer.add_space(20)

# Modelin katmanlarının kapsamını PDF'e kaydedin (threshold coverage)
threshold_value = 0

data = [
    [
        "Layer index",
        "Activation function",
        "Number of covered neurons",
        "Number of total neurons",
        "Coverage value",
    ]
]

pdf_writer.add_text(
    "Threshold Coverage Values of Layers (TH = " + str(threshold_value) + ")",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows threshold coverage values about the '"
    + model_info["name"]
    + "' model's all layers. NOTE: The threshold coverage value of a layer is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the "
    + str(input_idx)
    + "th input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

coverage_values_of_layers = driver.get_th_coverage_of_layers(input_idx, threshold_value)
for idx, (
    layer_index,
    activation_fn,
    num_of_covered_neurons,
    total_neurons,
    coverage,
) in enumerate(coverage_values_of_layers):
    data.append(
        [
            "Layer " + str(layer_index),
            str(activation_fn),
            str(num_of_covered_neurons),
            str(total_neurons),
            f"{coverage * 100:.2f}%",
        ]
    )

num_of_covered_neurons, total_neurons, coverage = driver.get_th_coverage_of_model(
    input_idx, threshold_value
)
data.append(
    [
        "All model",
        "-",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)
pdf_writer.add_table(data)

pdf_writer.save()
