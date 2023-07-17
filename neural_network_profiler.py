import torch
import torch.nn as nn
import numpy as np
from activation_functions import ActivationFunctions
from torchviz import make_dot
import copy

"""
Overview: The purpose of this class is to record the general information of the 
analyzed model. This information includes information such as the total number 
of parameters found in the model, the total number of layers, which activation 
function is used in each layer, what are the values after the activation function 
for the neurons in that layer.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class NeuralNetworkProfiler:
    @staticmethod
    def get_layer_names(model):
        """
        This function is no longer in use. It can be removed later!

        This function records and returns information about which activation function is used in the layers in the model.

        Args:
            model (torch.nn.Module): Model whose layer information will be recorded.
        Returns:
            model_layers_arr (list): List of the layers in the model.
        """
        model_layers_arr = []

        for name, layer in model._modules.items():
            if is_iterable(layer):
                print(dir(layer))
                model_layers_arr.extend(NeuralNetworkProfiler.get_layer_names(layer))
            else:
                model_layers_arr.append(layer)

        return model_layers_arr

    @staticmethod
    def run_model_with_array(input, model_layers_arr):
        """
        This function is no longer in use. It can be removed later!
        """
        for layer in model_layers_arr:
            layer_result = layer(input)
            print(layer_result)
            input = layer_result

    @staticmethod
    def get_model_info(model):
        """
        This function records and returns general information about the model. This
        information includes information such as the total number of parameters
        found in the model, name of the model, total number of layers etc.

        Args:
            model (torch.nn.Module): Model whose general information will be recorded.
        Returns:
            model_info (dict): Dictionary that contains the general information of the model.
        """
        model_info = {}

        # Modelin adını ve türünü kaydetme
        model_info["name"] = type(model).__name__

        # Modelin parametre sayısını kaydetme
        total_params = sum(p.numel() for p in model.parameters())
        model_info["total_params"] = total_params

        # Modelin katmanlarını ve katman sayısını kaydetme
        layers = []
        num_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Module):
                layers.append(name)
                num_layers += 1
        model_info["layers"] = layers
        model_info["num_layers"] = num_layers

        # Modelin optimizer ve kayıp fonksiyonunu kaydetme (varsa)
        if hasattr(model, "optimizer"):
            model_info["optimizer"] = type(model.optimizer).__name__
        if hasattr(model, "criterion"):
            model_info["criterion"] = type(model.criterion).__name__

        return model_info

    @staticmethod
    def get_activation_info(model, test_input):
        """
        This function records and returns information about which activation function is
        used in the layers in the model. It also records the values after the activation
        function for the neurons in that layer.

        Args:
            model (torch.nn.Module): Model whose activation information will be recorded.
            test_input (numpy.ndarray): Input that will be used to run the model.
        Returns:
            activation_info (dict): Dictionary that contains the activation information of the model.
        """
        model = copy.deepcopy(model)

        # Aktivasyon bilgilerini depolamak için bir dictionary oluşturun
        activation_info = {}

        # Hook fonksiyonunu tanımlayın
        def save_activation(consecutive_index, layer_index):
            def hook(module, input, output):
                activation_info[str(consecutive_index)] = {
                    "layer_index": layer_index,
                    "act_func": module.__class__.__name__,
                    # "before_act_func_values": input[0].detach().cpu().numpy(),
                    "after_act_func_values": output.detach().cpu().numpy(),
                }

            return hook

        activation_functions = ActivationFunctions.get_activation_functions()
        # Aktivasyon fonksiyonlarına hook'u ekleyin
        consecutive_index = 0
        for layer_index, module in enumerate(model.modules()):
            if any(
                isinstance(module, activation_function)
                for activation_function in activation_functions
            ):
                module.register_forward_hook(
                    save_activation(
                        consecutive_index,
                        layer_index,
                    )
                )
                consecutive_index += 1

        # Girdi tensörünü oluşturun
        input_tensor = torch.tensor(test_input)

        # İleri geçişi yapın
        model.eval()  # Modeli değerlendirme moduna getirin
        with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırakın
            output = model(input_tensor)

        return activation_info

    @staticmethod
    def get_activation_infos_for_multiple_inputs(model, test_inputs):
        """
        This function is prepared to run the function named 'get_activation_info' for more
        than one input. It performs the same functions for more than one input.

        Args:
            model (torch.nn.Module): Model whose activation information will be recorded.
            test_inputs (list): List of the inputs that will be used to run the model.
        Returns:
            activation_info_for_multiple_inputs (list): List of the activation information of the model.
        """
        activation_info_for_multiple_inputs = []

        for test_input in test_inputs:
            activation_info = NeuralNetworkProfiler.get_activation_info(
                model, test_input
            )
            activation_info_for_multiple_inputs.append(activation_info)

        return activation_info_for_multiple_inputs

    @staticmethod
    def get_model_architecture_dict_of_input(input, model_layers_arr):
        """
        This function is no longer in use. It can be removed later!
        This function is the non-generic version of the 'get_activation_info' function.
        No longer used!
        """
        model_architecture_dict_of_input = {}  # initializing the model architecture
        index = 0

        for idx, layer in zip(
            range(len(model_layers_arr)), model_layers_arr
        ):  # looping on loaded model activation functions
            layer = model_layers_arr[
                idx
            ]  # execute an operation on the loaded model activation functions array
            layer_result = layer(input)  # save result on the variable

            for activation_function in ActivationFunctions.get_activation_functions():
                if isinstance(
                    layer, activation_function
                ):  # find a model class defination for example (relu, sigmoid, tanh...)
                    layer_props = {
                        "layer_index": idx,
                        "act_func": layer,
                        "before_act_func_values": input,
                        "after_act_func_values": layer_result,
                    }
                    model_architecture_dict_of_input[str(index)] = layer_props

                    index = index + 1

            input = layer_result

        return model_architecture_dict_of_input  # return the model architecture

    @staticmethod
    def get_model_architecture_dicts_of_inputs(inputs, model_layers_arr):
        """
        # This function is no longer in use. It can be removed later!
        """
        model_architecture_dicts_of_inputs = []

        for input in inputs:
            model_architecture_dict_of_input = (
                NeuralNetworkProfiler.get_model_architecture_dict_of_input(
                    input, model_layers_arr
                )
            )
            model_architecture_dicts_of_inputs.append(model_architecture_dict_of_input)

        return model_architecture_dicts_of_inputs  # return the model architecture

    @staticmethod
    def get_counter_dict_of_model(model, test_input):
        """
        The purpose of this function is to create a dictionary structure to record
        the number of times each neuron is activated in case the model is run for
        more than one test input.

        Args:
            model (torch.nn.Module): Model whose activation information will be recorded.
            test_input (numpy.ndarray): Input that will be used to run the model.
        Returns:
            counter_dict (dict): Dictionary to prepare to save the number of times each neuron is activated.
        """
        model = copy.deepcopy(model)

        # Aktivasyon bilgilerini depolamak için bir dictionary oluşturun
        activation_info = {}

        # Hook fonksiyonunu tanımlayın
        def save_activation(consecutive_index, layer_index):
            def hook(module, input, output):
                activation_info[str(consecutive_index)] = {
                    "layer_index": layer_index,
                    "act_func": module.__class__.__name__,
                    "how_many_times_activated": np.zeros_like(
                        output.detach().cpu().numpy()
                    ),
                }

            return hook

        activation_functions = ActivationFunctions.get_activation_functions()
        # Aktivasyon fonksiyonlarına hook'u ekleyin
        consecutive_index = 0
        for layer_index, module in enumerate(model.modules()):
            if any(
                isinstance(module, activation_function)
                for activation_function in activation_functions
            ):
                module.register_forward_hook(
                    save_activation(
                        consecutive_index,
                        layer_index,
                    )
                )
                consecutive_index += 1

        # Girdi tensörünü oluşturun
        input_tensor = torch.tensor(test_input)

        # İleri geçişi yapın
        model.eval()  # Modeli değerlendirme moduna getirin
        with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırakın
            output = model(input_tensor)

        return activation_info

    @staticmethod
    def visualize_model(model, test_input):
        make_dot(model(test_input).mean(), params=dict(model.named_parameters()))
