import os
import os.path
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from analyzer import Analyzer
from image_loader import ImageLoader


class NeuralNetwork(nn.Module):
    # Here we define the model. If the model is to be loaded from the local
    # machine, the model definition is needed.
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def analyze_neural_network():
    # This function shows how to analyze any model. The operations found here could
    # also be defined directly within the 'main()' function. However, such a structure
    # was preferred for the readability of the code. Loading the model in the function,
    # defining the transform variable in order to make the data set suitable, and defining
    # the parameter values to be used during the analysis processes are performed.
    model = NeuralNetwork()
    model = torch.load("./model_fashion_1.pth")  # Modelin yüklenmesi

    # The definition of the 'transform' variable may differ from model to model and
    # dataset to dataset. Therefore, it must be defined separately for each model.
    # The transform variable found here is defined for the MNIST dataset. In order
    # to define the 'transform' variable suitable for the data set and the model,
    # the documentation of the model should be examined and the recommended variable
    # should be used.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )  # Defining the transform variable to optimize the dataset
    image_loader = ImageLoader("./test", transform)  # Loading the dataset

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
    ]  # Defining parameter values to be used during analysis processes

    analyze = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
    )  # Calling the Analyzer class
    analyze.analyze()


def analyze_maxvit():
    model = models.maxvit_t(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Reshape image size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet18():
    model = models.resnet18(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet34():
    model = models.resnet34(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet50():
    model = models.resnet50(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet101():
    model = models.resnet101(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet152():
    model = models.resnet152(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_alexnet():
    model = models.alexnet(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def main():
    # analyze_neural_network()
    # analyze_maxvit()
    # analyze_alexnet()
    # analyze_resnet18()
    # analyze_resnet34()
    # analyze_resnet50()
    # analyze_resnet101()
    analyze_resnet152()


if __name__ == "__main__":
    main()
