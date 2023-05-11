from math import inf
import torch
import numpy as np
from model_architecture_utils import ModelArchitectureUtils
from coverage_utils import CoverageUtils


class Coverage:
    @staticmethod
    def get_neuron_coverage_for_single_layer(layer):
        mean_of_layer = torch.mean(layer)

        num_of_covered_neurons_in_layer = len(layer[layer > mean_of_layer])
        total_neurons_in_layer = len(layer[layer > -inf])

        return (
            num_of_covered_neurons_in_layer,
            total_neurons_in_layer,
            num_of_covered_neurons_in_layer / total_neurons_in_layer,
        )

    @staticmethod
    def get_neuron_coverage_for_all_layers(layers):
        num_of_covered_neurons = 0
        total_neurons = 0

        for layer in layers:
            (
                num_of_covered_neurons_in_layer,
                total_neurons_in_layer,
                _,
            ) = Coverage.get_neuron_coverage_for_single_layer(layer)
            num_of_covered_neurons += num_of_covered_neurons_in_layer
            total_neurons += total_neurons_in_layer

        return (
            num_of_covered_neurons,
            total_neurons,
            (num_of_covered_neurons / total_neurons),
        )

    @staticmethod
    def get_threshold_coverage_for_single_layer(layer, threshold_value=0.75):
        num_of_covered_neurons_in_layer = len(layer[layer > threshold_value])
        total_neurons_in_layer = len(layer[layer > -inf])

        return (
            num_of_covered_neurons_in_layer,
            total_neurons_in_layer,
            num_of_covered_neurons_in_layer / total_neurons_in_layer,
        )

    @staticmethod
    def get_threshold_coverage_for_all_layers(layers, threshold_value=0.75):
        num_of_covered_neurons = 0
        total_neurons = 0

        for layer in layers:
            (
                num_of_covered_neurons,
                total_neurons,
                _,
            ) = Coverage.get_threshold_coverage_for_single_layer(layer)
            num_of_covered_neurons += num_of_covered_neurons
            total_neurons += total_neurons

        return (
            num_of_covered_neurons,
            total_neurons,
            (num_of_covered_neurons / total_neurons),
        )

    @staticmethod
    # Definition of fn: This function sequentially traverses all layers for 2 different test
    # inputs and detects the neurons that are jointly activated for two different test inputs.
    # It shows True if the same neuron of the same layer is active for both test inputs,
    # False if one is active for the other, False if both are inactive.
    # The 'results_of_layer' variable in this function specifies the 'after activation functions'
    # values obtained as a result of different test inputs of all layers.
    def compare_two_test_inputs_with_th_value(
        layers_of_first_input, layers_of_second_input, threshold_value=0.75
    ):
        result_bool_arr_of_layers = []

        for layer_of_first_input, layer_of_second_input in zip(
            layers_of_first_input, layers_of_second_input
        ):
            bool_tensor_first = layer_of_first_input > threshold_value
            bool_tensor_second = layer_of_second_input > threshold_value

            result_bool_arr_of_layer = torch.logical_and(
                bool_tensor_first, bool_tensor_second
            )
            result_bool_arr_of_layers.append(result_bool_arr_of_layer)

        return result_bool_arr_of_layers

    @staticmethod
    # Definition of fn: This function accepts the 'after activation functions' values of the layers
    # as a variable named 'layers_of_test_inputs' for each of the multiple test inputs as a parameter.
    # This function browses each test input as a parameter and finds an average coverage value for each
    # layer of each test input and counts the number of neurons in the relevant layer above this average
    # coverage value. These operations are performed separately for each layer of each test input. As a
    # result, it returns the total value of how many neurons were found in total for all test inputs, and
    # as a result, how many neurons were above the average value of each layer.
    def get_average_neuron_coverage_with_multiple_inputs(layers_of_test_inputs):
        total_num_of_covered_neurons = 0
        total_num_of_neurons = 0

        for layers_of_test_input in layers_of_test_inputs:
            (
                num_of_covered_neurons,
                num_of_neurons,
                _,
            ) = Coverage.get_neuron_coverage_for_all_layers(layers_of_test_input)
            total_num_of_covered_neurons += num_of_covered_neurons
            total_num_of_neurons += num_of_neurons

        return (
            total_num_of_covered_neurons,
            total_num_of_neurons,
            total_num_of_covered_neurons / total_num_of_neurons,
        )

    @staticmethod
    # Definition of fn: This function takes only the 'after activation function' values of the relevant layer
    # for multiple test inputs, the index of the neuron dealing with in the relevant layer, and the threshold
    # value parameters as parameters.
    # It visits all test inputs for the relevant neuron in the relevant layer and calculates how many times the
    # relevant neuron has been activated for these test inputs.
    def how_many_times_specific_neuron_activated(
        model_architecture_dicts, layer_index, neuron_index, threshold_value=0.75
    ):
        total_num_of_times_activated = 0

        for model_architecture_dict in model_architecture_dicts:
            after_values_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(
                    model_architecture_dict
                )
            )
            layer = after_values_all_layers[layer_index]
            neuron_value = layer[0][neuron_index]

            if neuron_value > threshold_value:
                total_num_of_times_activated += 1

        return total_num_of_times_activated

    @staticmethod
    # Definition of fn: This function runs each of the given test inputs and checks how many of the given x amount
    # of test inputs are above the threshold value for each neuron on the deep neural network.
    # The variable named 'counter_dict' records how many times each neuron in each layer is active.
    def how_many_times_neurons_activated(
        counter_dict, model_architecture_dicts, threshold_value=0.75
    ):
        temp_counter_dict = counter_dict.copy()

        for model_architecture_dict in model_architecture_dicts:
            after_values_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(
                    model_architecture_dict
                )
            )

            for layer_index, layer in enumerate(after_values_all_layers):
                for neuron_index, neuron_value in enumerate(layer[0]):
                    if neuron_value > threshold_value:
                        temp_counter_dict[str(layer_index)]["how_many_times_activated"][
                            neuron_index
                        ] += 1

        return temp_counter_dict

    @staticmethod
    def get_sign_coverage(
        model_architecture_dict_for_tI, model_architecture_dict_for_tII
    ):
        covered_neurons = 0
        total_neurons = 0

        for k in range(
            len(model_architecture_dict_for_tI)
        ):  # k specifies the layer index
            after_act_fn_values = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    model_architecture_dict_for_tI, k
                )[0]
            )
            for neuron_index in range(
                len(after_act_fn_values)
            ):  # neuron_index specifies the i value
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    model_architecture_dict_for_tI,
                    model_architecture_dict_for_tII,
                ):
                    covered_neurons = covered_neurons + 1

                total_neurons = total_neurons + 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_sign_sign_coverage(
        model_architecture_dict_for_tI, model_architecture_dict_for_tII
    ):
        covered_neurons = 0
        total_neurons = 0

        for k in range(
            len(model_architecture_dict_for_tI)
        ):  # k specifies the layer index, looping on the first test input
            if (
                k == len(model_architecture_dict_for_tI) - 1
            ):  # check if the two test input are in the same index
                break

            after_act_fn_values = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    model_architecture_dict_for_tI, k
                )[0]
            )
            after_act_fn_values_for_consecutive_layer = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    model_architecture_dict_for_tI, k + 1
                )[0]
            )

            for neuron_index in range(
                len(after_act_fn_values)
            ):  # neuron_index specifies the i value
                for neuron_index_for_consecutive_layer in range(
                    len(after_act_fn_values_for_consecutive_layer)
                ):  # neuron_index_for_consecutive_layer specifies the j value
                    if CoverageUtils.is_there_sign_sign_change(
                        k,
                        neuron_index,
                        neuron_index_for_consecutive_layer,
                        model_architecture_dict_for_tI,
                        model_architecture_dict_for_tII,
                    ):
                        covered_neurons = covered_neurons + 1

                    total_neurons = total_neurons + 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons
