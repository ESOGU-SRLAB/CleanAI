import torch
import torch.nn as nn
import numpy as np
from activation_functions import ActivationFunctions
from torchviz import make_dot


class NeuralNetworkProfiler:
    @staticmethod
    def get_layer_names(model):
        model_layers_arr = []

        for name, layer in model._modules.items():
            if isinstance(layer, nn.Sequential):
                model_layers_arr.extend(NeuralNetworkProfiler.get_layer_names(layer))
            else:
                model_layers_arr.append(layer)

        return model_layers_arr

    @staticmethod
    def run_model_with_array(input, model_layers_arr):
        for layer in model_layers_arr:
            layer_result = layer(input)
            print(layer_result)
            input = layer_result

    @staticmethod
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
    def get_counter_dict_of_model(model_layers_arr):
        counter_dict_of_model = {}  # initializing the model architecture
        index = 0

        for idx, layer in zip(range(len(model_layers_arr)), model_layers_arr):
            layer = model_layers_arr[idx]

            for activation_function in ActivationFunctions.get_activation_functions():
                if isinstance(
                    layer, activation_function
                ):  # find a model class defination for example (relu, sigmoid, tanh...)
                    layer_props = {
                        "layer_index": idx,
                        "act_func": layer,
                        "how_many_times_activated": [0] * out_features,
                    }

                    counter_dict_of_model[str(index)] = layer_props
                    index = index + 1

                elif isinstance(layer, nn.Linear):
                    out_features = layer.out_features

        return counter_dict_of_model

    @staticmethod
    def visualize_model(model, test_input):
        make_dot(model(test_input).mean(), params=dict(model.named_parameters()))
