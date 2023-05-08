from model import NeuralNetwork, load_model, make_prediction
from train_model import NeuralNetworkTrainer
from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST

training_data = FashionMNIST(
root="data",
train=True,
download=True,
transform=ToTensor()
)

test_data = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model_name = 'model.pth'
model = load_model(model_name)

dataset = Dataset(training_data, test_data)
train_dataloader, test_dataloader = dataset.get_data_loaders()

if model is not None:
    print(model)
else:
    print('Model does not exist')
    model = NeuralNetwork()
    trainer = NeuralNetworkTrainer(model)
    trainer.train_and_save(train_dataloader, test_dataloader, learning_rate=1e-3, epochs=5, model_name = model_name)
    print(model)

test_input_with_label = dataset.get_random_input_with_label()
print([test_input_with_label])

make_prediction(model, test_input_with_label)


