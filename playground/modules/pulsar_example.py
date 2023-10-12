import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from zeta.nn.modules.pulsar import PulsarNew as Pulsar


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


# Extend the dataset loading to include a validation set
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)

# Validation function
def validate(model, val_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Benchmarking
results = {}

for name, act in activations.items():
    train_times = []
    val_accuracies = []
    
    # Multiple runs for reliability
    for run in range(3):
        model = NeuralNetwork(act)
        start_time = time.time()
        train(model, train_loader, epochs=5)
        end_time = time.time()
        
        train_times.append(end_time - start_time)
        val_accuracies.append(validate(model, val_loader))

    avg_train_time = sum(train_times) / len(train_times)
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    model_size = sum(p.numel() for p in model.parameters())
    
    results[name] = {
        "Avg Training Time": avg_train_time,
        "Avg Validation Accuracy": avg_val_accuracy,
        "Model Size (Params)": model_size
    }

# Print Results
for name, metrics in results.items():
    print(f"---- {name} ----")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    print("\n")
