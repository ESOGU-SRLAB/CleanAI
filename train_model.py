import torch
from torch import nn

class NeuralNetworkTrainer:
    def __init__(self, model):
        self.model = model

    def train_loop(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_and_save(self, train_dataloader, test_dataloader, learning_rate=1e-3, batch_size=64, epochs=5, model_name='model.pth'):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_dataloader, loss_fn, optimizer)
            self.test_loop(test_dataloader, loss_fn)
        print("Done!")

        torch.save(self.model, model_name)

    @classmethod
    def load_model(cls, model_name='model.pth'):
        model = torch.load(model_name)
        return cls(model)
