from enum import Enum
import torch.nn as nn


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ActivationFunctions(ExtendedEnum):
    RELU = nn.ReLU
    SIGMOID = nn.Sigmoid
    TANH = nn.Tanh
    LEAKY_RELU = nn.LeakyReLU
    ELU = nn.ELU
    RELU6 = nn.ReLU6
    # SELU = nn.SELU
    # CELU = nn.CELU
    # GLU = nn.GLU
    # GELU = nn.GELU

    def get_activation_functions():  # this function return all specified activation function as a enum.
        return ActivationFunctions.list()
