from dataset import Dataset
from custom_dataset import CustomDataset
from coverage import Coverage
from neural_network_profiler import NeuralNetworkProfiler
from model_architecture_utils import ModelArchitectureUtils

class Driver:
    def __init__(self, model, custom_dataset) -> None:
        self.model = model
        self.custom_dataset = custom_dataset

    def get_model_informations(self):
        return NeuralNetworkProfiler.get_model_info(self.model)
        
    def get_coverage_of_layer(self, input_idx, layer_index):
        sample, label = self.custom_dataset[input_idx]
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info, layer_index
            )
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_neuron_coverage_for_single_layer(layer_after_values)

        return (num_of_covered_neurons, total_neurons, coverage)
    
    def get_coverage_of_layers(self, input_idx):
        result = []
        sample, label = self.custom_dataset[input_idx]
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)

        for layer_index in range(len(activation_info)):
            layer_after_values = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info, layer_index
                )
            )

            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = Coverage.get_neuron_coverage_for_single_layer(layer_after_values)
            result.append((num_of_covered_neurons, total_neurons, coverage))
        
        return result
    
    def get_coverage_of_model(self, input_idx):
        sample, label = self.custom_dataset[input_idx]
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = (
            ModelArchitectureUtils.get_after_values_for_all_layers(activation_info)
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_neuron_coverage_for_all_layers(layer_after_values)

        return (num_of_covered_neurons, total_neurons, coverage)

    
