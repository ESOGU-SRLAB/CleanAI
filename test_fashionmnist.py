import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

from driver import Driver
from custom_dataset import CustomDataset
from print_utils import PDFTableGenerator
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
            nn.Linear(28*28, 512),
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
pdf_file_name = "analysis_fashionmnist.pdf"
pdf_generator = PDFTableGenerator(pdf_file_name)

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
table_title = MODEL_INFORMATIONS_TITLE()
table_description = MODEL_INFORMATIONS_DESCRIPTION(model_info["name"])
column_headers = MODEL_INFORMATIONS_COL_HEADERS()
row_headers = MODEL_INFORMATIONS_ROW_HEADERS()
table_data = MODEL_INFORMATIONS_TABLE_DATA(
    model_info["name"], model_info["total_params"], model_info["num_layers"]
)

pdf_generator.add_table(table_data, column_headers, row_headers, table_title, table_description)
pdf_generator

# Modelin katmanlarının kapsamını hesaplayın
coverage_values_of_layers = driver.get_coverage_of_layers(input_idx)
for (num_of_covered_neurons, total_neurons, coverage) in coverage_values_of_layers:
    pdf_generator.add_table(MODEL_COVERAGE_TABLE_DATA, MODEL_COVERAGE_COL_HEADERS, MODEL_COVERAGE_ROW_HEADERS, MODEL_COVERAGE_TITLE, MODEL_COVERAGE_DESCRIPTION(model_info["name"]))

pdf_generator.generate_pdf()