from coverage import Coverage
from coverage_utils import CoverageUtils
from neural_network_profiler import NeuralNetworkProfiler
from model_architecture_utils import ModelArchitectureUtils


class Driver:
    def __init__(self, model) -> None:
        self.model = model

    def get_model_informations(self):
        return NeuralNetworkProfiler.get_model_info(self.model)

    def get_coverage_of_layer(self, sample, layer_index):
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
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)

        (tknc_sum, num_of_selected_neurons, mean_top_k) = Coverage.TKNC(
            activation_info, top_k
        )

        return (tknc_sum, num_of_selected_neurons, mean_top_k)

    def get_nbc_coverage_of_model(self, samples_for_bound, sample_for_nbc):
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
        activation_info = NeuralNetworkProfiler.get_activation_info(self.model, sample)
        counter_arr, total_neurons = Coverage.MNC(node_intervals, activation_info)
        return counter_arr, total_neurons
