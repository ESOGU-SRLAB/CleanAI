from torch import nn
import torch

label_arr = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    @staticmethod
    def make_prediction(model, input_arr):
        with torch.no_grad():
            for input in input_arr:
                image = input[0]
                label = input[1]
                pred = model(image)
                print(f"Predicted: {label_arr[pred.argmax(1)[0].item()]}, Actual: {label_arr[label]}")