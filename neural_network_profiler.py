import torch
import torch.nn as nn
import numpy as np
from activation_functions import ActivationFunctions
from torchviz import make_dot
import copy


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class NeuralNetworkProfiler:
    @staticmethod
    # This function is no longer in use. It can be removed later!
    def get_layer_names(model):
        model_layers_arr = []

        for name, layer in model._modules.items():
            if is_iterable(layer):
                print(dir(layer))
                model_layers_arr.extend(NeuralNetworkProfiler.get_layer_names(layer))
            else:
                model_layers_arr.append(layer)

        return model_layers_arr

    @staticmethod
    # This function is no longer in use. It can be removed later!
    def run_model_with_array(input, model_layers_arr):
        for layer in model_layers_arr:
            layer_result = layer(input)
            print(layer_result)
            input = layer_result

    @staticmethod
    def get_model_info(model):
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
        activation_info_for_multiple_inputs = []

        for test_input in test_inputs:
            activation_info = NeuralNetworkProfiler.get_activation_info(
                model, test_input
            )
            activation_info_for_multiple_inputs.append(activation_info)

        return activation_info_for_multiple_inputs

    @staticmethod
    # This function is no longer in use. It can be removed later!
    def get_model_architecture_dict_of_input(input, model_layers_arr):
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
    # This function is no longer in use. It can be removed later!
    def get_model_architecture_dicts_of_inputs(inputs, model_layers_arr):
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
