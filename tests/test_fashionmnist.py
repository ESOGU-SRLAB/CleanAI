import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

from driver import Driver
from custom_dataset import CustomDataset
from print_utils import PDFWriter
from image_loader import ImageLoader
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


# FashionMNIST modelini yükleme
model = torch.load("./model_fashion_1.pth")

# FashionMNIST veri setini yükleme
image_loader = ImageLoader("./test")
sample, random_image_name = image_loader.get_random_input()
sample_II, random_image_name_II = image_loader.get_random_input()

samples = []
how_many_samples = 2
for i in range(how_many_samples):
    sample, random_image_name = image_loader.get_random_input()
    samples.append(sample)


# FashionMNIST sınıf etiketleri
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

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
    "Coverage Values of Layers (For Only One Input)",
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

# Modelin katmanlarının kapsamını PDF'e kaydedin (neuron coverage for multiple inputs)
data = [
    [
        "Layer index",
        "Number of covered neurons",
        "Number of total neurons",
        "Coverage value",
    ]
]
(
    num_of_covered_neurons,
    total_neurons,
    coverage,
) = driver.get_average_coverage_of_model(samples)
pdf_writer.add_text(
    "Coverage Values of Layers (For Multiple Inputs) " + str(len(samples)) + " Inputs",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows coverage values for multiple inputs about the '"
    + model_info["name"]
    + "' model. The values in the table below, it was formed as a result of giving the '"
    + str(len(samples))
    + "' inputs in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)
data.append(
    [
        "All model",
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

pdf_writer.add_space(20)
# Modelin Sign Coverage ve Value Coverage kapsamını PDF'e kaydedin (sign coverage & value coverage)
threshold_value = 0.75

data = [
    [
        "Coverage Metric",
        "Number of covered neurons",
        "Number of total neurons",
        "Coverage value",
    ]
]

pdf_writer.add_text(
    "Sign Coverage and Value Coverage (TH = "
    + str(threshold_value)
    + ") Values of Model",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows Sign Coverage and Value Coverage values of the '"
    + model_info["name"]
    + "' model. Sign Coverage: When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented. Value Coverage: When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' and '"
    + random_image_name_II
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

num_of_covered_neurons, total_neurons, coverage = driver.get_sign_coverage_of_model(
    sample, sample_II
)
data.append(
    [
        "Sign Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)


num_of_covered_neurons, total_neurons, coverage = driver.get_value_coverage_of_model(
    sample, sample_II, threshold_value
)
data.append(
    [
        "Value Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)

pdf_writer.add_space(5)
pdf_writer.add_table(data)

pdf_writer.add_space(20)
# Modelin SS, SV, VS, VV kapsamını PDF'e kaydedin (SS, SV, VS, VV coverage)
threshold_value = -5

data = [
    [
        "Coverage Metric",
        "Number of covered neuron pairs",
        "Number of total neuron pairs",
        "Coverage value",
    ]
]

pdf_writer.add_text(
    "SS, SV, VS and VV Coverage (TH = " + str(threshold_value) + ") Values of Model",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows Sign-Sign Coverage, Sign-Value Coverage, Value-Sign Coverage and Value-Value Coverage values of the '"
    + model_info["name"]
    + "' model. Sign-Sign Coverage: When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented. Value Coverage: When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' and '"
    + random_image_name_II
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

num_of_covered_neurons, total_neurons, coverage = driver.get_ss_coverage_of_model(
    sample, sample_II
)
data.append(
    [
        "Sign-Sign Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)


num_of_covered_neurons, total_neurons, coverage = driver.get_sv_coverage_of_model(
    sample, sample_II, threshold_value
)
data.append(
    [
        "Sign-Value Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)

num_of_covered_neurons, total_neurons, coverage = driver.get_vs_coverage_of_model(
    sample, sample_II, threshold_value
)
data.append(
    [
        "Value-Sign Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)

num_of_covered_neurons, total_neurons, coverage = driver.get_vv_coverage_of_model(
    sample, sample_II, threshold_value
)
data.append(
    [
        "Value-Value Coverage",
        str(num_of_covered_neurons),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)

pdf_writer.add_space(5)
pdf_writer.add_table(data)

pdf_writer.add_space(20)
# Modelin TKNC kapsamını PDF'e kaydedin (TKNC coverage)
top_k_value = 5

data = [
    [
        "Coverage Metric",
        "TKNC Sum",
        "Number of Selected Neurons",
        "Mean of Top-K Neurons",
    ]
]

pdf_writer.add_text(
    "Top-K Neuron Coverage (K = " + str(top_k_value) + ") Value of Model",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows Top-K Neuron Coverage value of the '"
    + model_info["name"]
    + "' model. Top-K Neuron Coverage (TKNC) is a metric used to evaluate the activation patterns and coverage of neurons in a deep neural network (DNN). It measures the percentage of neurons that are activated for a given set of input samples. The idea behind TKNC is to assess how well a set of input samples can activate different neurons in the network. How is it calculated? TKNC travels through all layers on a model one by one and ranks the neuron values of each layer in order from largest to smallest. Then it takes k neurons in each layer and adds it to a list. It then creates a value called 'TKNC Sum', which represents the sum of neurons in this list. The 'Number of Selected Neurons' value shows how many neurons were selected on the whole model as a result of k neurons from each layer. The 'Mean of Top-K Neurons' value shows the ratio of the 'TKNC Sum' value to the 'Number of Selected Neurons' value. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

tknc_sum, num_of_selected_neurons, mean_top_k = driver.get_tknc_coverage_of_model(
    sample, top_k_value
)
data.append(
    [
        "Top-K Neuron Coverage",
        str(tknc_sum),
        str(num_of_selected_neurons),
        str(mean_top_k),
    ]
)

pdf_writer.add_space(5)
pdf_writer.add_table(data)

pdf_writer.add_space(20)
# Modelin NBC kapsamını PDF'e kaydedin (NBC coverage)

data = [
    [
        "Coverage Metric",
        "NBC Counter",
        "Number of Total Neurons",
        "Neuron Boundary Coverage",
    ]
]

pdf_writer.add_text(
    "Neuron Boundary Coverage Value of Model (For " + str(len(samples)) + " Inputs)",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows Neuron Boundary Coverage value of the '"
    + model_info["name"]
    + "' model. Neuron Boundary Coverage (NBC) is a metric used to evaluate the coverage of decision boundaries in a deep neural network (DNN). It measures the percentage of decision boundaries in the network that have been activated or crossed by the input samples. How is it calculated? NBC receives a random set of inputs from the user, and as a result of these inputs, it determines the maximum and minimum interval value for each layer. Then, for the input data to be checked, it is checked whether each neuron belonging to each layer is within the maximum and minimum range of this layer. If it is within this range, the 'NBC Counter' value is increased by one. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

nbc_counter, total_neurons, coverage = driver.get_nbc_coverage_of_model(samples, sample)
data.append(
    [
        "Neuron Boundary Coverage",
        str(nbc_counter),
        str(total_neurons),
        f"{coverage * 100:.2f}%",
    ]
)

pdf_writer.add_space(5)
pdf_writer.add_table(data)

pdf_writer.add_space(20)
# Modelin MNC kapsamını PDF'e kaydedin (MNC coverage)

data = [
    [
        "Threshold Intervals",
        "MNC Counter",
        "Number of Total Neurons",
        "Multisection Neuron Coverage",
    ]
]

pdf_writer.add_text(
    "Multisection Neuron Coverage Value of Model",
    font_size=16,
    is_bold=True,
)
pdf_writer.add_text(
    f"The table below shows Multisection Neuron Coverage value of the '"
    + model_info["name"]
    + "' model. Multisection Neuron Coverage (MNC) specifically focuses on assessing the coverage of individual neurons within the model. The goal of MNC is to evaluate the degree to which the decisions made by individual neurons have been exercised by the test cases. It helps identify potential shortcomings in the model's behavior and reveal areas that may require further testing. It provides the user with the information of how many neurons are found according to the threshold value ranges given by the user. How is it calculated? The MNC receives threshold ranges from the user. Then, it evaluates all the neurons on the model and checks whether each neuron is within these threshold ranges. If the corresponding neuron is within this threshold value, it increases the 'MNC Counter' value found for the relevant range by one. The 'Multisection Neuron Coverage' value is the ratio of the 'MNC Counter' value to the number of all neurons on the model. The values in the table below, it was formed as a result of giving the '"
    + random_image_name
    + "' input in the data set to the model.",
    font_size=14,
    is_bold=False,
)
pdf_writer.add_space(5)

node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
counter_arr, total_neurons = driver.get_mnc_coverage_of_model(sample, node_intervals)

for i, res in enumerate(counter_arr):
    data.append(
        [
            f"{node_intervals[i][0]} - {node_intervals[i][1]}",
            str(counter_arr[i]),
            str(total_neurons),
            f"{counter_arr[i] / total_neurons * 100:.2f}%",
        ]
    )

pdf_writer.add_space(5)
pdf_writer.add_table(data)

pdf_writer.save()
