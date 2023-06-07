import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import requests

from neural_network_profiler import NeuralNetworkProfiler
from coverage import Coverage
from coverage_utils import CoverageUtils
from model_architecture_utils import ModelArchitectureUtils
from print_utils import PrintUtils, PDFTableGenerator
from driver import Driver
from definitions import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_convnets_processing_utils"
)

# Modeli kaydetme
"""
model = models.resnet50(pretrained=True)
model.eval()  # Modeli eğitim modundan çıkarır ve değerleri dondurur

torch.save(model.state_dict(), "resnet50_model.pth")
"""

# Kaydedilen modeli yükleme

model = models.resnet50()
model.load_state_dict(torch.load("resnet50_model.pth"))
model.eval()

# ---------------------------------------

uris = [
    "https://images.unsplash.com/photo-1560807707-8cc77767d783",
    "https://images.unsplash.com/photo-1684495498026-3419b55bdbac",
    "https://images.unsplash.com/photo-1684707973359-8f9c35e99afd?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=687&q=80",
]

batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in uris]).to(device)

with torch.no_grad():
    output = torch.nn.functional.softmax(model(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256, 256), Image.ANTIALIAS)
    # plt.imshow(img)
    # plt.show()
    # print(result)

# ---------------------------------------
# This section allows the general information of the model to be saved and presented to the user.

pdf_generator = PDFTableGenerator("ResNET50_Analysis.pdf")
model_info = NeuralNetworkProfiler.get_model_info(model)

table_title = MODEL_INFORMATIONS_TITLE()
table_description = MODEL_INFORMATIONS_DESCRIPTION(model_info["name"])
column_headers = MODEL_INFORMATIONS_COL_HEADERS()
row_headers = MODEL_INFORMATIONS_ROW_HEADERS()
table_data = MODEL_INFORMATIONS_TABLE_DATA(
    model_info["name"], model_info["total_params"], model_info["num_layers"]
)

pdf_generator.add_table(
    table_data, column_headers, row_headers, table_title, table_description
)
pdf_generator.generate_pdf()


activation_info_for_tI = NeuralNetworkProfiler.get_activation_info(
    model, batch[0].unsqueeze(0)
)
activation_info_for_tII = NeuralNetworkProfiler.get_activation_info(
    model, batch[1].unsqueeze(0)
)
activation_info_for_tIII = NeuralNetworkProfiler.get_activation_info(
    model, batch[2].unsqueeze(0)
)

last_layer_after_values_for_tI = (
    ModelArchitectureUtils.get_after_values_for_specific_layer(
        activation_info_for_tI, len(activation_info_for_tI) - 1
    )
)
all_layers_after_values_for_tI = ModelArchitectureUtils.get_after_values_for_all_layers(
    activation_info_for_tI
)

all_layers_after_values_for_tII = (
    ModelArchitectureUtils.get_after_values_for_all_layers(activation_info_for_tII)
)

# print(f"---------------------------------------")
# print(f"activation_info_for_tI: {activation_info_for_tI}\n")
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"last_layer_after_values_for_tI: {last_layer_after_values_for_tI}\n")
# print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(
    f"Coverage.get_neuron_coverage_for_single_layer (last_layer (for test input I)):\n"
)
(
    num_of_covered_neurons,
    total_neurons,
    coverage,
) = Coverage.get_neuron_coverage_for_single_layer(last_layer_after_values_for_tI)
PrintUtils.print_table(
    ["Number of covered neurons", "Total number of neurons", "Coverage"],
    ["Last layer values (input I)"],
    [[num_of_covered_neurons, total_neurons, "{:.2f}".format(coverage * 100) + "%"]],
)


print(f"\n---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.get_neuron_coverage_for_all_layers (for test input I):\n")
(
    num_of_covered_neurons,
    total_neurons,
    coverage,
) = Coverage.get_neuron_coverage_for_all_layers(all_layers_after_values_for_tI)
PrintUtils.print_table(
    ["Number of covered neurons", "Total number of neurons", "Coverage"],
    ["All layers values (input I)"],
    [[num_of_covered_neurons, total_neurons, "{:.2f}".format(coverage * 100) + "%"]],
)
SaveAnalysis.save_analysis(
    ["Number of covered neurons", "Total number of neurons", "Coverage"],
    "All layers values (input I)",
    [num_of_covered_neurons, total_neurons, "{:.2f}".format(coverage * 100) + "%"],
)

print(f"\n---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.get_neuron_coverage_for_all_layers (for test input II):\n")
(
    num_of_covered_neurons,
    total_neurons,
    coverage,
) = Coverage.get_neuron_coverage_for_all_layers(all_layers_after_values_for_tII)
PrintUtils.print_table(
    ["Number of covered neurons", "Total number of neurons", "Coverage"],
    ["All layers values (input II)"],
    [
        [
            num_of_covered_neurons,
            total_neurons,
            "{:.2f}".format(coverage * 100) + "%",
        ]
    ],
)
SaveAnalysis.save_analysis(
    ["Number of covered neurons", "Total number of neurons", "Coverage"],
    "All layers values (input II)",
    [
        num_of_covered_neurons,
        total_neurons,
        "{:.2f}".format(coverage * 100) + "%",
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(
    f"Coverage.get_threshold_coverage_for_single_layer (last_layer (for test input I)):\n"
)
(
    num_of_th_covered_neurons,
    total_neurons,
    th_coverage,
) = Coverage.get_threshold_coverage_for_single_layer(
    last_layer_after_values_for_tI, 0.1
)
PrintUtils.print_table(
    [
        "Threshold value",
        "Number of covered neurons (> threshold)",
        "Total number of neurons",
        "Coverage",
    ],
    ["Last layer values (input I)"],
    [
        [
            0.1,
            num_of_th_covered_neurons,
            total_neurons,
            "{:.2f}".format(th_coverage * 100) + "%",
        ]
    ],
)
SaveAnalysis.save_analysis(
    [
        "Threshold value",
        "Number of covered neurons (> threshold)",
        "Total number of neurons",
        "Coverage",
    ],
    "Last layer values (input I)",
    [
        0.1,
        num_of_th_covered_neurons,
        total_neurons,
        "{:.2f}".format(th_coverage * 100) + "%",
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")

print(f"Coverage.get_threshold_coverage_for_all_layers (for test input I):\n")
(
    num_of_th_covered_neurons,
    total_neurons,
    th_coverage,
) = Coverage.get_threshold_coverage_for_all_layers(all_layers_after_values_for_tI, 0.1)
PrintUtils.print_table(
    [
        "Threshold value",
        "Number of covered neurons (> threshold)",
        "Total number of neurons",
        "Coverage",
    ],
    ["All layers values (input I)"],
    [
        [
            0.1,
            num_of_th_covered_neurons,
            total_neurons,
            "{:.2f}".format(th_coverage * 100) + "%",
        ]
    ],
)
SaveAnalysis.save_analysis(
    [
        "Threshold value",
        "Number of covered neurons (> threshold)",
        "Total number of neurons",
        "Coverage",
    ],
    "All layers values (input I)",
    [
        0.1,
        num_of_th_covered_neurons,
        total_neurons,
        "{:.2f}".format(th_coverage * 100) + "%",
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(
    f"Coverage.compare_two_test_inputs_with_th_value (for test input I & test input II):\n"
)
result = Coverage.compare_two_test_inputs_with_th_value(
    all_layers_after_values_for_tI, all_layers_after_values_for_tII, 0.1
)
# print(f"Result: {results}")

print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.get_average_neuron_coverage_with_multiple_inputs:\n")
activation_infos_arr = [activation_info_for_tI, activation_info_for_tII]
layers_of_inputs = ModelArchitectureUtils.get_after_values_for_multiple_inputs(
    activation_infos_arr
)
(
    num_of_covered_neurons,
    total_neurons,
    coverage,
) = Coverage.get_average_neuron_coverage_with_multiple_inputs(layers_of_inputs)
PrintUtils.print_table(
    [
        "How many inputs are there?",
        "Number of covered neurons",
        "Total number of neurons",
        "Coverage",
    ],
    ["All layers values"],
    [
        [
            len(activation_infos_arr),
            num_of_covered_neurons,
            total_neurons,
            "{:.2f}".format(coverage * 100) + "%",
        ]
    ],
)
SaveAnalysis.save_analysis(
    [
        "How many inputs are there?",
        "Number of covered neurons",
        "Total number of neurons",
        "Coverage",
    ],
    "All layers values",
    [
        len(activation_infos_arr),
        num_of_covered_neurons,
        total_neurons,
        "{:.2f}".format(coverage * 100) + "%",
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(f"NeuralNetworkProfiler.get_counter_dict_of_model:\n")
counter_dict_of_model = NeuralNetworkProfiler.get_counter_dict_of_model(
    model, batch[2].unsqueeze(0)
)
# print(f"counter_dict_of_model: {counter_dict_of_model}")
# NOT: Bu kısıma çıktı olarak bir nöronların kaçının verdilerin test girdi sayısının yüzde 25/50/75 sinden fazlasında
# aktifleştirildiği gibi değerleri gösteren bir tablo eklenebilir.
print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.how_many_times_neurons_activated:\n")
# hw_many_times_activated = Coverage.how_many_times_neurons_activated(
#     counter_dict_of_model, [activation_info_for_tI, activation_info_for_tII], 0.1
# )
# print(f"hw_many_times_activated: {hw_many_times_activated}")
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.get_sign_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_sign_coverage(
#     activation_info_for_tI, activation_info_for_tII
# )
# PrintUtils.print_table(
#     [
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Sign coverage"],
#     [
#         [
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.get_value_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_value_coverage(
#     activation_info_for_tI, activation_info_for_tII, 0.15
# )
# PrintUtils.print_table(
#     [
#         "Threshold value",
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Value coverage"],
#     [
#         [
#             0.15,
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

# --------------------------------------- SS-SV-VS-VV in our model ---------------------------------------
# model = torch.load("model.pth")
# print(model)
# training_data = FashionMNIST(
#     root="data", train=True, download=True, transform=ToTensor()
# )

# test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())


# dataset = Dataset(training_data, test_data)
# activation_info_for_tI = NeuralNetworkProfiler.get_activation_info(
#     model, dataset.get_random_input()
# )
# activation_info_for_tII = NeuralNetworkProfiler.get_activation_info(
#     model, dataset.get_random_input()
# )

# print(f"---------------------------------------")
# print(f"Coverage.get_sign_sign_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_sign_sign_coverage(
#     activation_info_for_tI, activation_info_for_tII
# )
# PrintUtils.print_table(
#     [
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Sign-Sign coverage"],
#     [
#         [
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.get_value_value_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_value_value_coverage(
#     activation_info_for_tI, activation_info_for_tII, 0.01
# )
# PrintUtils.print_table(
#     [
#         "Threshold value",
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Value-Value coverage"],
#     [
#         [
#             0.01,
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.get_sign_value_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_sign_value_coverage(
#     activation_info_for_tI, activation_info_for_tII, 0.01
# )
# PrintUtils.print_table(
#     [
#         "Threshold value",
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Sign-Value coverage"],
#     [
#         [
#             0.01,
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

# print(f"---------------------------------------")
# print(f"Coverage.get_value_sign_coverage:\n")
# (covered_neurons, total_neurons, coverage) = Coverage.get_value_sign_coverage(
#     activation_info_for_tI, activation_info_for_tII, 0.01
# )
# PrintUtils.print_table(
#     [
#         "Threshold value",
#         "Number of covered neurons",
#         "Total number of neurons",
#         "Coverage",
#     ],
#     ["Value-Sign coverage"],
#     [
#         [
#             0.01,
#             covered_neurons,
#             total_neurons,
#             "{:.2f}".format(coverage * 100) + "%",
#         ]
#     ],
# )
# print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.TKNC:\n")
(tknc_sum, num_of_selected_neurons, mean_top_k) = Coverage.TKNC(
    activation_info_for_tI, 5
)
PrintUtils.print_table(
    [
        "K value",
        "Sum of Top-K neurons",
        "Number of selected neurons",
        "Mean of Top-K neurons",
    ],
    ["Top-K coverage"],
    [
        [
            5,
            "{:.2f}".format(tknc_sum),
            num_of_selected_neurons,
            "{:.2f}".format(mean_top_k),
        ]
    ],
)
SaveAnalysis.save_analysis(
    [
        "K value",
        "Sum of Top-K neurons",
        "Number of selected neurons",
        "Mean of Top-K neurons",
    ],
    "Top-K coverage",
    [
        5,
        "{:.2f}".format(tknc_sum),
        num_of_selected_neurons,
        "{:.2f}".format(mean_top_k),
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.NBC:\n")
bound_dict = CoverageUtils.get_bounds_for_layers(
    [activation_info_for_tI, activation_info_for_tII]
)
(nbc_counter, total_neurons, nbc_coverage) = Coverage.NBC(
    activation_info_for_tIII, bound_dict
)
PrintUtils.print_table(
    [
        "NBC counter",
        "Number of total neurons",
        "Coverage",
    ],
    ["Neuron boundary coverage"],
    [
        [
            nbc_counter,
            total_neurons,
            "{:.2f}".format(nbc_coverage * 100) + "%",
        ]
    ],
)
SaveAnalysis.save_analysis(
    [
        "NBC counter",
        "Number of total neurons",
        "Coverage",
    ],
    "Neuron boundary coverage",
    [
        nbc_counter,
        total_neurons,
        "{:.2f}".format(nbc_coverage * 100) + "%",
    ],
)
print(f"---------------------------------------\n")

print(f"---------------------------------------")
print(f"Coverage.MNC:\n")
res_arr = Coverage.MNC([(0.5, 0.8), (0.9, 1.5)], activation_info_for_tI)
print(f"counter_arr: {res_arr}")
print(f"---------------------------------------\n")