from math import inf
import torch
import tensorflow as tf
import numpy as np

from model_architecture_utils import ModelArchitectureUtils

"""
Overview: This class provides the auxiliary functions required for 
the 'Coverage' class when calculating coverage metrics.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class CoverageUtils:
    @staticmethod
    def is_there_sign_change(k, i, activation_info_for_tI, activation_info_for_tII):
        """
        Checks if there is a sign change for the neuron with index i in layer k

        Args:
            k (int): Layer index
            i (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
        Returns:
            bool: True if there is a sign change, False otherwise
        """
        neuron_value_for_tI = ModelArchitectureUtils.get_neuron_value_from_after_values(
            activation_info_for_tI, k, i
        )
        neuron_value_for_tII = (
            ModelArchitectureUtils.get_neuron_value_from_after_values(
                activation_info_for_tII, k, i
            )
        )

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
        """
        Checks if there is a value change for the neuron with index i in layer k

        Args:
            k (int): Layer index
            i (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
            threshold_value (float): Threshold value for value change
        Returns:
            bool: True if there is a value change, False otherwise
        """
        neuron_value_for_tI = ModelArchitectureUtils.get_neuron_value_from_after_values(
            activation_info_for_tI, k, i
        )
        neuron_value_for_tII = (
            ModelArchitectureUtils.get_neuron_value_from_after_values(
                activation_info_for_tII, k, i
            )
        )

        if abs(neuron_value_for_tI - neuron_value_for_tII) > threshold_value:
            return True
        else:
            return False

    @staticmethod
    def is_there_sign_sign_change(
        k, i, j, activation_info_for_tI, activation_info_for_tII
    ):
        """
        Checks if there is a sign-sign change for the neuron with index i in layer k
        and neuron with index j in layer k+1

        Args:
            k (int): Layer index
            i (int): Neuron index
            j (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
        Returns:
            flag (bool): True if there is a sign-sign change, False otherwise
        """
        flag = True

        after_act_func_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info_for_tI, k
            )
        )

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
        """
        Checks if there is a value-value change for the neuron with index i in layer k
        and neuron with index j in layer k+1

        Args:
            k (int): Layer index
            i (int): Neuron index
            j (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
        Returns:
            flag (bool): True if there is a value-value change, False otherwise
        """
        flag = True

        after_act_func_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info_for_tI, k
            )
        )

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
        """
        Checks if there is a sign-value change for the neuron with index i in layer k
        and neuron with index j in layer k+1

        Args:
            k (int): Layer index
            i (int): Neuron index
            j (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
        Returns:
            flag (bool): True if there is a sign-value change, False otherwise
        """
        flag = True

        after_act_func_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info_for_tI, k
            )
        )

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
        """
        Checks if there is a value-sign change for the neuron with index i in layer k
        and neuron with index j in layer k+1

        Args:
            k (int): Layer index
            i (int): Neuron index
            j (int): Neuron index
            activation_info_for_tI (dict): Activation information for test input I
            activation_info_for_tII (dict): Activation information for test input II
        Returns:
            flag (bool): True if there is a value-sign change, False otherwise
        """
        flag = True

        after_act_func_values_for_tI = (
            ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info_for_tI, k
            )
        )

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
    def calc_bounds(activation_infos, layer_idx):
        """
        This function accepts the layer values and the layer index created
        as a result of the input values given as parameters. The function finds the minimum
        and maximum neuron value of the relevant layer in the given layer index for each
        input value. As a result, the function presents the maximum and minimum neuron values
        of the layers related (specific, in the layer index specified as a parameter) to the
        user as a result of each input value.

        Args:
            activation_infos (list): List of activation information for each test input
            layer_idx (int): Layer index

        Returns:
            min, max (tuple): Minimum and maximum neuron values of the layer
        """
        min = inf
        max = -inf

        for activation_info in activation_infos:
            after_act_func_values_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(activation_info)
            )

            np_min = np.min(after_act_func_values_all_layers[layer_idx])
            np_max = np.max(after_act_func_values_all_layers[layer_idx])

            min = min if min < np_min else np_min
            max = max if max > np_max else np_max

        return min, max

    @staticmethod
    def get_bounds_for_layers(activation_infos):
        """
        This function navigates between the layer values of the input
        values given as parameters and returns the minimum and maximum neuron values for
        each layer. There are functions such as checking whether a neuron has undergone
        sign change or value change under two different inputs and calculating the average
        value of the relevant layer.

        Args:
            activation_infos (list): List of activation information for each test input

        Returns:
            bound_dict (dict): Dictionary of minimum and maximum neuron values for each layer
        """
        bound_dict = {}

        for layer_idx in range(len(activation_infos[0])):
            min_bound_of_layer, max_bound_of_layer = CoverageUtils.calc_bounds(
                activation_infos, layer_idx
            )
            prop = {"min_bound": min_bound_of_layer, "max_bound": max_bound_of_layer}
            bound_dict[str(layer_idx)] = prop

        return bound_dict

    @staticmethod
    def count_elements_above_threshold(arr, threshold):
        """
        This function counts the number of elements in the array that are greater than the threshold value.

        Args:
            arr (numpy.ndarray): Array of values
            threshold (float): Threshold value

        Returns:
            count (int): Number of elements in the array that are greater than the threshold value
        """
        flattened_arr = arr.flatten()  # Diziyi tek boyutlu hale getirme
        count = np.sum(
            flattened_arr > threshold
        )  # Eşik değerinden büyük elemanların sayısını hesaplama
        return count

    @staticmethod
    def calculate_mean(arr):
        """
        This function calculates the mean of all elements in the array.

        Args:
            arr (numpy.ndarray): Array of values

        Returns:
            mean (float): Mean of all elements in the array
        """
        flattened_arr = arr.flatten()  # Diziyi tek boyutlu hale getirme
        mean = np.mean(flattened_arr)  # Tüm elemanların ortalamasını hesaplama
        return mean