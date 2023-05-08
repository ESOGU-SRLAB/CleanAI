from math import inf
from logging import exception
from math import inf

class NeuronCoverage:
    @staticmethod
    def get_neuron_coverage_for_single_layer(layer):
        num_of_covered_neurons_in_layer = len(layer[layer > 0])
        total_neurons_in_layer = len(layer[layer > -inf])
        return (num_of_covered_neurons_in_layer, total_neurons_in_layer, num_of_covered_neurons_in_layer / total_neurons_in_layer)

    @staticmethod
    def get_neuron_coverage_for_all_layers(layers):
        num_of_covered_neurons = 0
        total_neurons = 0
        for layer in layers:
            num_of_covered_neurons_in_layer, total_neurons_in_layer, _ = NeuronCoverage.get_neuron_coverage_for_single_layer(layer)
            num_of_covered_neurons += num_of_covered_neurons_in_layer
            total_neurons += total_neurons_in_layer

        return (num_of_covered_neurons, total_neurons, (num_of_covered_neurons / total_neurons))
    
    @staticmethod    
    def get_threshold_coverage_for_single_layer(layer, threshold_value = 0.75):
        try:
            if threshold_value < 0 or threshold_value > 1:
                raise exception("threshold_value must be in range between 0 and 1")
            num_of_covered_neurons_in_layer = len(layer[layer > threshold_value])
            total_neurons_in_layer = len(layer[layer > -inf])
            return (num_of_covered_neurons_in_layer, total_neurons_in_layer, num_of_covered_neurons_in_layer / total_neurons_in_layer)
        except:
            print("threshold_value must be in range between 0 and 1")
    
    @staticmethod
    def get_threshold_coverage_for_all_layers(layers, threshold_value = 0.75):
        num_of_covered_neurons = 0
        total_neurons = 0
        try:
            if threshold_value < 0 or threshold_value > 1:
                raise exception("threshold_value must be in range between 0 and 1")
            for layer in layers:
                num_of_covered_neurons, total_neurons, _ = NeuronCoverage.get_threshold_coverage_for_single_layer(layer)
                num_of_covered_neurons += num_of_covered_neurons
                total_neurons += total_neurons

            return (num_of_covered_neurons, total_neurons, (num_of_covered_neurons / total_neurons))

        except:
            print("threshold_value must be in range between 0 and 1")
    
from math import inf
from logging import exception
from math import inf

class NeuronCoverage:
    @staticmethod
    def get_neuron_coverage_for_single_layer(layer):
        num_of_covered_neurons_in_layer = len(layer[layer > 0])
        total_neurons_in_layer = len(layer[layer > -inf])
        return (num_of_covered_neurons_in_layer, total_neurons_in_layer, num_of_covered_neurons_in_layer / total_neurons_in_layer)

    @staticmethod
    def get_neuron_coverage_for_all_layers(layers):
        num_of_covered_neurons = 0
        total_neurons = 0
        for layer in layers:
            num_of_covered_neurons_in_layer, total_neurons_in_layer, _ = NeuronCoverage.get_neuron_coverage_for_single_layer(layer)
            num_of_covered_neurons += num_of_covered_neurons_in_layer
            total_neurons += total_neurons_in_layer

        return (num_of_covered_neurons, total_neurons, (num_of_covered_neurons / total_neurons))
    
    @staticmethod    
    def get_threshold_coverage_for_single_layer(layer, threshold_value = 0.75):
        try:
            if threshold_value < 0 or threshold_value > 1:
                raise exception("threshold_value must be in range between 0 and 1")
            num_of_covered_neurons_in_layer = len(layer[layer > threshold_value])
            total_neurons_in_layer = len(layer[layer > -inf])
            return (num_of_covered_neurons_in_layer, total_neurons_in_layer, num_of_covered_neurons_in_layer / total_neurons_in_layer)
        except:
            print("threshold_value must be in range between 0 and 1")
    
    @staticmethod
    def get_threshold_coverage_for_all_layers(layers, threshold_value = 0.75):
        num_of_covered_neurons = 0
        total_neurons = 0
        try:
            if threshold_value < 0 or threshold_value > 1:
                raise exception("threshold_value must be in range between 0 and 1")
            for layer in layers:
                num_of_covered_neurons, total_neurons, _ = NeuronCoverage.get_threshold_coverage_for_single_layer(layer)
                num_of_covered_neurons += num_of_covered_neurons
                total_neurons += total_neurons

            return (num_of_covered_neurons, total_neurons, (num_of_covered_neurons / total_neurons))

        except:
            print("threshold_value must be in range between 0 and 1")
    
