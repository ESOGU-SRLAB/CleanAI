from model_architecture_utils import ModelArchitectureUtils
from neural_network_profiler import NeuralNetworkProfiler
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from math import inf
from coverage import Coverage
from coverage_utils import CoverageUtils
from model import load_model
from dataset import Dataset
from mini_model import load_mini_model

training_data = FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())


model = load_model("model.pth")
model_layers_arr = NeuralNetworkProfiler.get_layer_names(model)

dataset = Dataset(training_data, test_data)
random_input = dataset.get_random_input()

model_architecture_dict = NeuralNetworkProfiler.get_model_architecture_dict_of_input(
    random_input, model_layers_arr
)
after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
    model_architecture_dict, 0
)
after_values_all_layers = ModelArchitectureUtils.get_after_values_for_all_layers(
    model_architecture_dict
)

print(f"---------------------------------------\n")
print(f"Get neuron coverage for single layer: \n")
(
    _,
    num_of_covered_neurons,
    total_neurons,
    neuron_coverage,
) = Coverage.get_neuron_coverage_for_single_layer(after_values)
print(
    f"Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get neuron coverage for all layers: \n")
(
    num_of_covered_neurons,
    total_neurons,
    neuron_coverage,
) = Coverage.get_neuron_coverage_for_all_layers(after_values_all_layers)
print(
    f"Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get threshold coverage for single layer: \n")
(
    num_of_th_covered_neurons,
    total_neurons,
    threshold_coverage,
) = Coverage.get_threshold_coverage_for_single_layer(after_values, 0)
print(
    f"Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"Get threshold coverage for all layers: \n")
(
    num_of_th_covered_neurons,
    total_neurons,
    threshold_coverage,
) = Coverage.get_threshold_coverage_for_all_layers(after_values_all_layers)
print(
    f"Num of threshold covered neurons: {num_of_th_covered_neurons}\nNum of total neurons: {total_neurons}\nThreshold coverage: {threshold_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: compare_test_inputs_with_th_value: \n")
mini_model = load_mini_model()
model_layers_arr_for_mini = NeuralNetworkProfiler.get_layer_names(mini_model)
model_architecture_dict_for_mini_first = (
    NeuralNetworkProfiler.get_model_architecture_dict_of_input(
        dataset.get_random_input(), model_layers_arr_for_mini
    )
)

model_architecture_dict_for_mini_second = (
    NeuralNetworkProfiler.get_model_architecture_dict_of_input(
        dataset.get_random_input(), model_layers_arr_for_mini
    )
)

after_values_all_layers_for_mini_first = (
    ModelArchitectureUtils.get_after_values_for_all_layers(
        model_architecture_dict_for_mini_first
    )
)

after_values_all_layers_for_mini_second = (
    ModelArchitectureUtils.get_after_values_for_all_layers(
        model_architecture_dict_for_mini_second
    )
)

result = Coverage.compare_two_test_inputs_with_th_value(
    after_values_all_layers_for_mini_first, after_values_all_layers_for_mini_second, 0
)

print(f"Dict for T1: {model_architecture_dict_for_mini_first}")
print(f"Dict for T2: {model_architecture_dict_for_mini_second}")
print(f"Result: {result}")

print(f"---------------------------------------\n")


print(f"---------------------------------------\n")
print(f"TEST FN: get_average_threshold_coverage_with_multiple_inputs: \n")
random_50_inputs = dataset.get_random_inputs(50)
model_architecture_dicts_of_inputs = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_50_inputs, model_layers_arr
    )
)
after_values_all_layers_for_multiple_inputs = (
    ModelArchitectureUtils.get_after_values_for_multiple_inputs(
        model_architecture_dicts_of_inputs
    )
)
(
    num_of_covered_neurons,
    total_neurons,
    neuron_coverage,
) = Coverage.get_average_neuron_coverage_with_multiple_inputs(
    after_values_all_layers_for_multiple_inputs
)
print(
    f"Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: how_many_times_neurons_activated: \n")
random_5_inputs = dataset.get_random_inputs(5)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_5_inputs, model_layers_arr_for_mini
    )
)
counter_dict = NeuralNetworkProfiler.get_counter_dict_of_model(
    model_layers_arr_for_mini
)
print(
    f"model_architecture_dicts_of_inputs_for_mini: {model_architecture_dicts_of_inputs_for_mini}\n"
)
Coverage.how_many_times_neurons_activated(
    counter_dict, model_architecture_dicts_of_inputs_for_mini, 0.1
)
print(f"Counter dict: {counter_dict}")
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: how_many_times_specific_neuron_activated: \n")
print(
    f"model_architecture_dicts_of_inputs_for_mini: {model_architecture_dicts_of_inputs_for_mini}\n"
)
layer_idx = 0
neuron_idx = 0
th_value = 0.1
result = Coverage.how_many_times_specific_neuron_activated(
    model_architecture_dicts_of_inputs_for_mini, layer_idx, neuron_idx, th_value
)
print(
    f"{neuron_idx}. Neuron in Layer {layer_idx} activated {result} times while threshold value is {th_value}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_sign_sign_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_sign_sign_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_sign_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_sign_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_value_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_value_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_value_value_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_value_value_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_sign_value_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_sign_value_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: get_value_sing_coverage: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
covered_neurons, total_neurons, coverage = Coverage.get_value_sign_coverage(
    model_architecture_dicts_of_inputs_for_mini[0],
    model_architecture_dicts_of_inputs_for_mini[1],
)
print(
    f"Covered neurons: {covered_neurons}\nTotal neurons: {total_neurons}\nCoverage: {coverage}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: TKNC: \n")
random_2_inputs = dataset.get_random_inputs(2)
model_architecture_dicts_of_inputs_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_2_inputs, model_layers_arr_for_mini
    )
)
tknc_value = Coverage.TKNC(model_architecture_dicts_of_inputs_for_mini[0], 2)
print(f"TKNC values for model {tknc_value}")
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: NBC: \n")
random_50_inputs = dataset.get_random_inputs(50)
rand_input = dataset.get_random_input()
model_architecture_dict_of_input = (
    NeuralNetworkProfiler.get_model_architecture_dict_of_input(
        rand_input, model_layers_arr
    )
)
model_architecture_dicts_of_inputs = (
    NeuralNetworkProfiler.get_model_architecture_dicts_of_inputs(
        random_50_inputs, model_layers_arr
    )
)
bound_dict = CoverageUtils.get_bounds_for_layers(model_architecture_dicts_of_inputs)
nbc_counter, total_neurons, nbc_coverege = Coverage.NBC(
    model_architecture_dict_of_input, bound_dict
)
print(
    f"NBC counter: {nbc_counter}\nTotal neurons: {total_neurons}\nNBC coverage: {nbc_coverege}"
)
print(f"---------------------------------------\n")

print(f"---------------------------------------\n")
print(f"TEST FN: MBC: \n")
node_intervals = [(-0.35, 0.35), (0.35, 0.55)]
rand_input = dataset.get_random_input()
model_architecture_dict_of_input_for_mini = (
    NeuralNetworkProfiler.get_model_architecture_dict_of_input(
        rand_input, model_layers_arr_for_mini
    )
)

result_mnc = Coverage.MNC(node_intervals, model_architecture_dict_of_input_for_mini)

print(f"MNC result percentages: {result_mnc}")
print(f"---------------------------------------\n")
