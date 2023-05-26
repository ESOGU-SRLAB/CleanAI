from math import inf
import torch
import numpy as np
from model_architecture_utils import ModelArchitectureUtils
from coverage_utils import CoverageUtils
from neural_network_profiler import NeuralNetworkProfiler


class Coverage:
    @staticmethod
    def get_neuron_coverage_for_single_layer(layer):
        mean_of_layer = CoverageUtils.calculate_mean(layer)

        num_of_covered_neurons_in_layer = CoverageUtils.count_elements_above_threshold(
            layer, mean_of_layer
        )
        total_neurons_in_layer = CoverageUtils.count_elements_above_threshold(
            layer, -inf
        )

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
        num_of_covered_neurons_in_layer = CoverageUtils.count_elements_above_threshold(
            layer, threshold_value
        )
        total_neurons_in_layer = CoverageUtils.count_elements_above_threshold(
            layer, -inf
        )

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
                num_of_covered_neurons_in_layer,
                total_neurons_in_layer,
                _,
            ) = Coverage.get_threshold_coverage_for_single_layer(layer, threshold_value)
            num_of_covered_neurons += num_of_covered_neurons_in_layer
            total_neurons += total_neurons_in_layer

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

            result_bool_arr_of_layer = np.logical_and(
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
    # This function is no longer in use. It can be removed later!

    # Definition of fn: This function takes only the 'after activation function' values of the relevant layer
    # for multiple test inputs, the index of the neuron dealing with in the relevant layer, and the threshold
    # value parameters as parameters.
    # It visits all test inputs for the relevant neuron in the relevant layer and calculates how many times the
    # relevant neuron has been activated for these test inputs.
    def how_many_times_specific_neuron_activated(
        activation_infos, layer_index, neuron_index, threshold_value=0.75
    ):
        total_num_of_times_activated = 0

        for activation_info in activation_infos:
            after_values_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(activation_info)
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
        counter_dict, activation_infos, threshold_value=0.75
    ):
        # counter_dict = NeuralNetworkProfiler.get_counter_dict_of_model(
        #     model, activation_infos[0]
        # )

        for activation_info in activation_infos:
            after_values_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(activation_info)
            )

            for layer_idx, layer in enumerate(after_values_all_layers):
                for neuron_idx, neuron_val in np.ndenumerate(layer):
                    if neuron_val > threshold_value:
                        counter_dict[str(layer_idx)]["how_many_times_activated"][
                            neuron_idx
                        ] += 1
            # for layer_index, layer in enumerate(after_values_all_layers):
            #     for neuron_index, neuron_value in enumerate(layer[0]):
            #         if neuron_value > threshold_value:
            #             temp_counter_dict[str(layer_index)]["how_many_times_activated"][
            #                 neuron_index
            #             ] += 1

        return counter_dict

    @staticmethod
    def get_sign_coverage(activation_info_for_tI, activation_info_for_tII):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            for neuron_idx, neuron_val in np.ndenumerate(layer):
                if CoverageUtils.is_there_sign_change(
                    layer_idx,
                    neuron_idx,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    covered_neurons += 1

                total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

        # for k in range(len(activation_info_for_tI)):  # k specifies the layer index
        #     after_act_fn_values = (
        #         ModelArchitectureUtils.get_after_values_for_specific_layer(
        #             activation_info_for_tI, k
        #         )[0]
        #     )
        #     for neuron_index in range(
        #         len(after_act_fn_values)
        #     ):  # neuron_index specifies the i value
        #         if CoverageUtils.is_there_sign_change(
        #             k,
        #             neuron_index,
        #             activation_info_for_tI,
        #             activation_info_for_tII,
        #         ):
        #             covered_neurons = covered_neurons + 1

        #         total_neurons = total_neurons + 1

        # return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_value_coverage(
        activation_info_for_tI, activation_info_for_tII, threshold_value=0.75
    ):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            for neuron_idx, neuron_val in np.ndenumerate(layer):
                if CoverageUtils.is_there_value_change(
                    layer_idx,
                    neuron_idx,
                    activation_info_for_tI,
                    activation_info_for_tII,
                    threshold_value,
                ):
                    covered_neurons += 1

                total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_sign_sign_coverage(activation_info_for_tI, activation_info_for_tII):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            if layer_idx == len(after_act_fn_values_for_tI) - 1:
                break

            after_act_fn_values_for_consecutive_layer = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info_for_tI, layer_idx + 1
                )
            )

            for neuron_idx, neuron_val in np.ndenumerate(layer):
                for (
                    neuron_idx_for_consecutive_layer,
                    neuron_val_for_consecutive_layer,
                ) in np.ndenumerate(after_act_fn_values_for_consecutive_layer):
                    if CoverageUtils.is_there_sign_sign_change(
                        layer_idx,
                        neuron_idx,
                        neuron_idx_for_consecutive_layer,
                        activation_info_for_tI,
                        activation_info_for_tII,
                    ):
                        covered_neurons += 1

                    total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_value_value_coverage(
        activation_info_for_tI, activation_info_for_tII, threshold_value=0.75
    ):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            if layer_idx == len(after_act_fn_values_for_tI) - 1:
                break

            after_act_fn_values_for_consecutive_layer = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info_for_tI, layer_idx + 1
                )
            )

            for neuron_idx, neuron_val in np.ndenumerate(layer):
                for (
                    neuron_idx_for_consecutive_layer,
                    neuron_val_for_consecutive_layer,
                ) in np.ndenumerate(after_act_fn_values_for_consecutive_layer):
                    if CoverageUtils.is_there_value_value_change(
                        layer_idx,
                        neuron_idx,
                        neuron_idx_for_consecutive_layer,
                        activation_info_for_tI,
                        activation_info_for_tII,
                        threshold_value,
                    ):
                        covered_neurons += 1

                    total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_sign_value_coverage(
        activation_info_for_tI, activation_info_for_tII, threshold_value=0.75
    ):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            if layer_idx == len(after_act_fn_values_for_tI) - 1:
                break

            after_act_fn_values_for_consecutive_layer = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info_for_tI, layer_idx + 1
                )
            )

            for neuron_idx, neuron_val in np.ndenumerate(layer):
                for (
                    neuron_idx_for_consecutive_layer,
                    neuron_val_for_consecutive_layer,
                ) in np.ndenumerate(after_act_fn_values_for_consecutive_layer):
                    if CoverageUtils.is_there_sign_value_change(
                        layer_idx,
                        neuron_idx,
                        neuron_idx_for_consecutive_layer,
                        activation_info_for_tI,
                        activation_info_for_tII,
                        threshold_value,
                    ):
                        covered_neurons += 1

                    total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    def get_value_sign_coverage(
        activation_info_for_tI, activation_info_for_tII, threshold_value=0.75
    ):
        covered_neurons = 0
        total_neurons = 0

        after_act_fn_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info_for_tI
            )
        )

        for layer_idx, layer in enumerate(after_act_fn_values_for_tI):
            if layer_idx == len(after_act_fn_values_for_tI) - 1:
                break

            after_act_fn_values_for_consecutive_layer = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info_for_tI, layer_idx + 1
                )
            )

            for neuron_idx, neuron_val in np.ndenumerate(layer):
                for (
                    neuron_idx_for_consecutive_layer,
                    neuron_val_for_consecutive_layer,
                ) in np.ndenumerate(after_act_fn_values_for_consecutive_layer):
                    if CoverageUtils.is_there_value_sign_change(
                        layer_idx,
                        neuron_idx,
                        neuron_idx_for_consecutive_layer,
                        activation_info_for_tI,
                        activation_info_for_tII,
                        threshold_value,
                    ):
                        covered_neurons += 1

                    total_neurons += 1

        return covered_neurons, total_neurons, covered_neurons / total_neurons

    @staticmethod
    # Definition of fn: This function navigates the layer values resulting from the input
    # value and takes the highest neuron value as the 'top_k' value given as a parameter
    # for each layer. Then, it adds the values of 'top_k' neurons with the highest neuron
    # values in each layer to a variable (it performs this operation by traveling through
    # all layers). Finally, it divides this 'sum' variable by the number of neurons selected
    # and presents an average value to the user.
    def TKNC(activation_info, top_k):
        tknc_sum = 0

        after_act_fn_values = ModelArchitectureUtils.get_after_values_for_all_layers(
            activation_info
        )

        for layer_idx, layer in enumerate(after_act_fn_values):
            sorted_layer = np.sort(layer, axis=None)[::-1]
            sorted_top_k = sorted_layer[:top_k]

            sum_top_k = np.sum(sorted_top_k)
            tknc_sum = sum_top_k + tknc_sum

        num_of_selected_neurons = top_k * len(after_act_fn_values)
        mean_top_k = tknc_sum / num_of_selected_neurons

        return (tknc_sum, num_of_selected_neurons, mean_top_k)

    @staticmethod
    # Definition of fn: This function takes neuron values and layer information on the model
    # as a result of one test input as a parameter. As the second parameter, it takes an
    # input where the maximum and minimum neuron values of each layer are kept.
    # The function of the function: for each neuron value, a comparison is made with the
    # maximum and minimum neuron values in the relevant layer. If the neuron value is not
    # between these maximum and minimum values, the variable named 'nbc_counter' is
    # increased by one. The function calculates the number of neurons that are not between
    # these minimum and maximum values and its ratio.
    def NBC(activation_info, bound_dict):
        nbc_counter = 0
        total_neurons = 0

        after_act_fn_values = ModelArchitectureUtils.get_after_values_for_all_layers(
            activation_info
        )

        for layer_idx, layer in enumerate(after_act_fn_values):
            for neuron_idx, neuron_val in np.ndenumerate(layer):
                if (
                    neuron_val < bound_dict[str(layer_idx)]["min_bound"]
                    or neuron_val > bound_dict[str(layer_idx)]["max_bound"]
                ):
                    nbc_counter = nbc_counter + 1
                total_neurons = total_neurons + 1

        return nbc_counter, total_neurons, nbc_counter / total_neurons

    @staticmethod
    # Definition of fn: This function takes a variable named 'node_intervals' as a parameter.
    # The type of this variable is an array, and each index of this array contains a tupple
    # of 'lower_bound' and 'upper_bound' pairs. For each tupple, index 0 is 'lower_bound'
    # and index 1 is 'upper_bound'. The function cycles through the neuron values of the
    # specific layer given as a parameter, and checks whether the relevant neuron value is
    # between these lower and upper limits. It checks for each pair of lower and upper bound
    # limits. As a result, it returns a counter array with the same size as how many pairs of
    # lower and upper intervals are in the 'node_intervals' array.
    def MNC_for_single_layer(node_intervals, activation_info, layer_index):
        m = len(node_intervals)
        result = [0] * m

        after_act_fn_values_for_layer = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info, layer_index
            )
        )

        for neuron_idx, neuron_val in np.ndenumerate(after_act_fn_values_for_layer):
            for i in range(m):
                interval = node_intervals[i]
                lower_bound = interval[0]
                upper_bound = interval[1]

                if neuron_val >= lower_bound and neuron_val <= upper_bound:
                    result[i] = result[i] + 1

        return result

    @staticmethod
    # Definition of fn: Applies the 'MNC_for_single_layer' function for all layers.
    def MNC(node_intervals, activation_info):
        res_arr = []

        after_act_fn_values = ModelArchitectureUtils.get_after_values_for_all_layers(
            activation_info
        )

        for layer_idx, layer in enumerate(after_act_fn_values):
            counter_arr = Coverage.MNC_for_single_layer(
                node_intervals, activation_info, layer_idx
            )

            for index in range(len(counter_arr)):
                counter_arr[index] = counter_arr[
                    index
                ] / CoverageUtils.count_elements_above_threshold(layer, -inf)

            res_arr.append(counter_arr)

        return res_arr
