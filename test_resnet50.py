import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.models as models

from driver import Driver
from custom_dataset import CustomDataset
from print_utils import PDFWriter
from image_loader import ImageLoader
from definitions import *

# ResNet-50 modelini yükleme
model = models.resnet50(pretrained=True)
model.eval()


# Analiz sonuçlarının kaydedileceği PDF dosyasının ismi
pdf_file_name = "Analysis_ResNET50.pdf"
pdf_writer = PDFWriter(pdf_file_name)
pdf_writer.add_text("CleanAI Analysis Report", font_size=20, is_bold=True)


# FashionMNIST veri setini yükleme
image_loader = ImageLoader("./resnet50_samples")
sample, random_image_name = image_loader.get_random_input()


# Tahmin edilen sınıf etiketi ve ilgili olasılığı
print(model(sample))

# Driver sınıfını kullanarak modeli ve veri setini kullanarak bir sürücü oluşturun
driver = Driver(model)

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
coverage_values_of_layers = driver.get_coverage_of_layers(sample)
pdf_writer.add_text(
    "Coverage Values of Layers",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows coverage values about the '"
    + model_info["name"]
    + "' model's all layers. The 'mean of layer' value shows the average of neurons in that layer. When calculating the number of covered neurons, this value is accepted as the threshold value for that layer. NOTE: The coverage value of a layer is the ratio of the number of covered neurons to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
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
            f"{mean_of_layer:.2f}",
        ]
    )

num_of_covered_neurons, total_neurons, coverage = driver.get_coverage_of_model(sample)
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
    + "' model's all layers. NOTE: The threshold coverage value of a layer is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

coverage_values_of_layers = driver.get_th_coverage_of_layers(sample, threshold_value)
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
    sample, threshold_value
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
threshold_value = 0.75

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
    + "' model's all layers. NOTE: The threshold coverage value of a layer is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

coverage_values_of_layers = driver.get_th_coverage_of_layers(sample, threshold_value)
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
    sample, threshold_value
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
