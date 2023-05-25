from math import inf
import torch
import tensorflow as tf
import numpy as np


class CoverageUtils:
    @staticmethod
    def is_there_sign_change(k, i, activation_info_for_tI, activation_info_for_tII):
        neuron_value_for_tI = activation_info_for_tI[str(k)]["after_act_func_values"][i]
        neuron_value_for_tII = activation_info_for_tII[str(k)]["after_act_func_values"][
            i
        ]

        if neuron_value_for_tI * neuron_value_for_tII <= 0:
            return True
        else:
            return False

    @staticmethod
    def is_there_value_change(
        k,
        i,
        activation_info_for_tI,
        activation_info_for_tII,
        threshold_value=0.05,
    ):
        neuron_value_for_tI = activation_info_for_tI[str(k)]["after_act_func_values"][i]
        neuron_value_for_tII = activation_info_for_tII[str(k)]["after_act_func_values"][
            i
        ]

        if abs(neuron_value_for_tI - neuron_value_for_tII) > threshold_value:
            return True
        else:
            return False

    @staticmethod
    def is_there_sign_sign_change(
        k, i, j, activation_info_for_tI, activation_info_for_tII
    ):
        flag = True

        after_act_func_values_for_tI = activation_info_for_tI[str(k)][
            "after_act_func_values"
        ]
        for neuron_index, neuron_value in np.ndenumerate(after_act_func_values_for_tI):
            if i == neuron_index:
                if not CoverageUtils.is_there_sign_change(
                    k,
                    i,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break
            else:
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break

        if not CoverageUtils.is_there_sign_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII
        ):
            flag = False

        return flag

    @staticmethod
    def is_there_value_value_change(
        k, i, j, activation_info_for_tI, activation_info_for_tII, threshold_value=0.05
    ):
        flag = True

        after_act_func_values_for_tI = activation_info_for_tI[str(k)][
            "after_act_func_values"
        ]

        for neuron_index, neuron_value in np.ndenumerate(after_act_func_values_for_tI):
            if i == neuron_index:
                if not CoverageUtils.is_there_value_change(
                    k,
                    i,
                    activation_info_for_tI,
                    activation_info_for_tII,
                    threshold_value,
                ):
                    flag = False
                    break
            else:
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break

        if not CoverageUtils.is_there_value_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII
        ):
            flag = False
        if CoverageUtils.is_there_sign_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII
        ):
            flag = False

        return flag

    @staticmethod
    def is_there_sign_value_change(
        k, i, j, activation_info_for_tI, activation_info_for_tII, threshold_value=0.05
    ):
        flag = True

        after_act_func_values_for_tI = activation_info_for_tI[str(k)][
            "after_act_func_values"
        ]

        for neuron_index, neuron_value in np.ndenumerate(after_act_func_values_for_tI):
            if i == neuron_index:
                if not CoverageUtils.is_there_sign_change(
                    k,
                    i,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break
            else:
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break

        if not CoverageUtils.is_there_value_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII, threshold_value
        ):
            flag = False

        if CoverageUtils.is_there_sign_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII
        ):
            flag = False

        return flag

    @staticmethod
    def is_there_value_sign_change(
        k, i, j, activation_info_for_tI, activation_info_for_tII, threshold_value=0.05
    ):
        flag = True

        after_act_func_values_for_tI = activation_info_for_tI[str(k)][
            "after_act_func_values"
        ]

        for neuron_index, neuron_value in np.ndenumerate(after_act_func_values_for_tI):
            if i == neuron_index:
                if not CoverageUtils.is_there_value_change(
                    k,
                    i,
                    activation_info_for_tI,
                    activation_info_for_tII,
                    threshold_value,
                ):
                    flag = False
                    break
            else:
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    activation_info_for_tI,
                    activation_info_for_tII,
                ):
                    flag = False
                    break

        if not CoverageUtils.is_there_sign_change(
            k + 1, j, activation_info_for_tI, activation_info_for_tII
        ):
            flag = False

        return flag

    @staticmethod
    # Definition of fn: This function accepts the layer values and the layer index created
    # as a result of the input values given as parameters. The function finds the minimum
    # and maximum neuron value of the relevant layer in the given layer index for each
    # input value. As a result, the function presents the maximum and minimum neuron values
    # of the layers related (specific, in the layer index specified as a parameter) to the
    # user as a result of each input value.
    def calc_bounds(activation_infos, layer_idx):
        min = inf
        max = -inf

        for activation_info in activation_infos:
            torch_min = torch.min(
                activation_info[str(layer_idx)]["after_act_func_values"][0]
            )
            torch_max = torch.max(
                activation_info[str(layer_idx)]["after_act_func_values"][0]
            )

            torch_min = torch_min.item()
            torch_max = torch_max.item()

            min = min if min < torch_min else torch_min
            max = max if max > torch_max else torch_max

        return min, max

    @staticmethod
    # Definition of fn: This function navigates between the layer values of the input
    # values given as parameters and returns the minimum and maximum neuron values for
    # each layer.
    def get_bounds_for_layers(activation_infos):
        bound_dict = {}

        for layer_idx in range(len(activation_infos[0])):
            min_bound_of_layer, max_bound_of_layer = CoverageUtils.calc_bounds(
                activation_infos, layer_idx
            )
            prop = {"min_bound": min_bound_of_layer, "max_bound": max_bound_of_layer}
            bound_dict[str(layer_idx)] = prop

        return bound_dict

    def count_elements_above_threshold(arr, threshold):
        flattened_arr = arr.flatten()  # Diziyi tek boyutlu hale getirme
        count = np.sum(
            flattened_arr > threshold
        )  # Eşik değerinden büyük elemanların sayısını hesaplama
        return count

    @staticmethod
    def calculate_mean(arr):
        flattened_arr = arr.flatten()  # Diziyi tek boyutlu hale getirme
        mean = np.mean(flattened_arr)  # Tüm elemanların ortalamasını hesaplama
        return mean
