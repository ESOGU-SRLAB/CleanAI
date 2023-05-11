from torch import nn
import torch
import os

label_arr = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Define model
class NeuralNetworkMini(nn.Module):
    def __init__(self):
        super(NeuralNetworkMini, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.ELU(),
            nn.Linear(3, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_mini_model():
    model = NeuralNetworkMini()
    return model
