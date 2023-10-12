import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from zeta.nn.modules.pulsar import Pulsar


# --- Neural Network Definition ---
class NeuralNetwork(nn.Module):
    def __init__(self, activation_function):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.activation = activation_function

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


# --- Dataset Preparation ---
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# --- Training Function ---
def train(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# --- Benchmarking ---
activations = {
    "ReLU": nn.ReLU(),
    "LogGamma": Pulsar(),
}

for name, act in activations.items():
    model = NeuralNetwork(act)
    start_time = time.time()
    train(model, train_loader)
    end_time = time.time()

    print(f"{name} - Training Time: {end_time - start_time:.2f} seconds")
