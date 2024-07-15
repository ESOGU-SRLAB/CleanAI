from coverage import Coverage
from coverage_utils import CoverageUtils
from neural_network_profiler import NeuralNetworkProfiler
from model_architecture_utils import ModelArchitectureUtils

"""
Overview: This class has been created in order to present the output values 
generated as a result of using the functions of the class named 'Coverage' 
to other classes in a proper format.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class Driver:
    def __init__(self, model) -> None:
        self.model = model

    def get_model_informations(self):
        """
        This function allows the general information of the given model to be returned.

        Args:
            None
        Returns:
            (dict): The general information of the given model.
        """
        return NeuralNetworkProfiler.get_model_info(self.model)

    def get_coverage_of_layer(self, sample, layer_index):
        """
        This function allows the coverage of the given layer to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            layer_index: The index of the layer to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the given layer.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
            activation_info, layer_index
        )

        (
            _,
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_neuron_coverage_for_single_layer(layer_after_values)

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_coverage_of_layers(self, sample):
        """
        This function allows the coverage of the layers to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the layers of model.
        """
        result = []
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)

        for layer_index in range(len(activation_info)):
            layer_after_values = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info, layer_index
                )
            )
            activation_fn = activation_info[str(layer_index)]["act_func"]
            layer_index = activation_info[str(layer_index)]["layer_index"]
            (
                mean_of_layer,
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = Coverage.get_neuron_coverage_for_single_layer(layer_after_values)
            result.append(
                (
                    layer_index,
                    activation_fn,
                    mean_of_layer,
                    num_of_covered_neurons,
                    total_neurons,
                    coverage,
                )
            )

        return result

    def get_coverage_of_model(self, sample):
        """
        This function allows the coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = ModelArchitectureUtils.get_after_values_for_all_layers(
            activation_info
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_neuron_coverage_for_all_layers(layer_after_values)

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_average_coverage_of_model(self, samples):
        """
        This function allows the average coverage of the given model to be returned.

        Args:
            samples: The samples to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the model.
        """
        num_of_covered_neurons = 0
        total_neurons = 0

        for sample in samples:
            activation_info = NeuralNetworkProfiler.get_activation_info(
                self.model, sample
            )
            layer_after_values = ModelArchitectureUtils.get_after_values_for_all_layers(
                activation_info
            )

            (
                num_of_covered_neurons_for_sample,
                total_neurons_for_sample,
                coverage_for_sample,
            ) = Coverage.get_neuron_coverage_for_all_layers(layer_after_values)

            num_of_covered_neurons += num_of_covered_neurons_for_sample
            total_neurons += total_neurons_for_sample

            del sample
            del activation_info

        coverage = num_of_covered_neurons / total_neurons
        return (num_of_covered_neurons, total_neurons, coverage)

    def get_th_coverage_of_layer(self, sample, layer_index, threshold_value):
        """
        This function allows the threshold coverage of the given layer to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            layer_index: The index of the layer to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
            activation_info, layer_index
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_threshold_coverage_for_single_layer(
            layer_after_values, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_th_coverage_of_layers(self, sample, threshold_value):
        """
        This function allows the threshold coverage of the layers to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the model.
        """
        result = []
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)

        for layer_index in range(len(activation_info)):
            layer_after_values = (
                ModelArchitectureUtils.get_after_values_for_specific_layer(
                    activation_info, layer_index
                )
            )
            activation_fn = activation_info[str(layer_index)]["act_func"]
            layer_index = activation_info[str(layer_index)]["layer_index"]
            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = Coverage.get_threshold_coverage_for_single_layer(
                layer_after_values, threshold_value
            )
            result.append(
                (
                    layer_index,
                    activation_fn,
                    num_of_covered_neurons,
                    total_neurons,
                    coverage,
                )
            )

        return result

    def get_th_coverage_of_model(self, sample, threshold_value):
        """
        This function allows the threshold coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            (tuple): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        layer_after_values = ModelArchitectureUtils.get_after_values_for_all_layers(
            activation_info
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_threshold_coverage_for_all_layers(
            layer_after_values, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_sign_coverage_of_model(self, sample, sample_II):
        """
        This function allows the sign coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_sign_coverage(activation_info, activation_info_II)

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_value_coverage_of_model(self, sample, sample_II, threshold_value):
        """
        This function allows the value coverage of the given model to be returned.

        Args:
            sample (): The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_value_coverage(
            activation_info, activation_info_II, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_ss_coverage_of_model(self, sample, sample_II):
        """
        This function allows the sign-sign coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_sign_sign_coverage(activation_info, activation_info_II)

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_sv_coverage_of_model(self, sample, sample_II, threshold_value):
        """
        This function allows the sign-value coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_sign_value_coverage(
            activation_info, activation_info_II, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_vv_coverage_of_model(self, sample, sample_II, threshold_value):
        """
        This function allows the value-value coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_value_value_coverage(
            activation_info, activation_info_II, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_vs_coverage_of_model(self, sample, sample_II, threshold_value):
        """
        This function allows the value-sign coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            sample_II: The sample to be used for the coverage calculation.
            threshold_value: The threshold value to be used for the coverage calculation.
        Returns:
            num_of_covered_neurons (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        activation_info_II = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_II
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = Coverage.get_value_sign_coverage(
            activation_info, activation_info_II, threshold_value
        )

        return (num_of_covered_neurons, total_neurons, coverage)

    def get_tknc_coverage_of_model(self, sample, top_k):
        """
        This function allows the top-k neuron coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            top_k: k value is the number of neurons to be selected.
        Returns:
            tknc_sum (int): The sum of the top-k neurons.
            num_of_selected_neurons (int): The number of selected neurons.
            mean_top_k (float): The mean of the top-k neurons.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)

        (tknc_sum, num_of_selected_neurons, mean_top_k) = Coverage.TKNC(
            activation_info, top_k
        )

        return (tknc_sum, num_of_selected_neurons, mean_top_k)

    def get_nbc_coverage_of_model(self, samples_for_bound, sample_for_nbc):
        """
        This function allows the neuron boundary coverage of the given model to be returned.

        Args:
            samples_for_bound: Selected input data for setting boundaries.
            sample_for_nbc: Input data from which to check whether it is within
                            the limit value ranges.
        Returns:
            nbc_counter (int): The number of covered neurons.
            total_neurons (int): The total number of neurons in the model.
            coverage (float): The coverage of the model.
        """
        activation_infos = []

        for sample in samples_for_bound:
            activation_infos.append(
                NeuralNetworkProfiler.get_activation_info(self.model, sample)
            )

        bound_dict = CoverageUtils.get_bounds_for_layers(activation_infos)
        activation_info = NeuralNetworkProfiler.get_activation_info(
            self.model, sample_for_nbc
        )

        (nbc_counter, total_neurons, coverage) = Coverage.NBC(
            activation_info, bound_dict
        )

        return (nbc_counter, total_neurons, coverage)

    def get_mnc_coverage_of_model(self, sample, node_intervals):
        """
        This function allows the multisection neuron coverage of the given model to be returned.

        Args:
            sample: The sample to be used for the coverage calculation.
            node_intervals: Specifies the coverage ranges.
        Returns:
            counter_arr (list): The number of covered neurons for each range.
            total_neurons (int): The total number of neurons in the model.
        """
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        counter_arr, total_neurons = Coverage.MNC(node_intervals, activation_info)
        return counter_arr, total_neurons
