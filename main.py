import os
import os.path
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from analyzer import Analyzer
from image_loader import ImageLoader


class NeuralNetwork(nn.Module):
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
    model = NeuralNetwork()
    model = torch.load("./model_fashion_1.pth")

    image_loader = ImageLoader("./test")

    how_many_samples = 5
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyze = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
    )
    analyze.analyze()


def analyze_maxvit():
    model = models.maxvit_t(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Görüntü boyutunu yeniden şekillendir
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 5
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
    )
    analyzer.analyze()


def main():
    # analyze_neural_network()
    analyze_maxvit()


if __name__ == "__main__":
    main()
