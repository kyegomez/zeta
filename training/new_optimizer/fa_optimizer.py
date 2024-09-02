import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from zeta.optim.all_new_optimizer import FastAdaptiveOptimizer
from torch.optim import Adam
import matplotlib.pyplot as plt
import time


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    return loss.item()


# Test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


# Main function to run the experiment
def run_experiment(optimizer_class, **optimizer_kwargs):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    epochs = 10
    train_losses = []
    test_losses = []
    test_accuracies = []
    training_time = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train(model, device, train_loader, optimizer, epoch)
        end_time = time.time()
        training_time += end_time - start_time

        test_loss, test_accuracy = test(model, device, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return train_losses, test_losses, test_accuracies, training_time


# Run experiments
optimizers = {
    "FastAdaptiveOptimizer": (
        FastAdaptiveOptimizer,
        {"lr": 0.001, "adaptive_lr": True, "warmup_steps": 1000},
    ),
    "Adam": (Adam, {"lr": 0.001}),
}

results = {}

for name, (opt_class, opt_kwargs) in optimizers.items():
    print(f"\nRunning experiment with {name}")
    train_losses, test_losses, test_accuracies, training_time = run_experiment(
        opt_class, **opt_kwargs
    )
    results[name] = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "training_time": training_time,
    }

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for name, data in results.items():
    plt.plot(data["train_losses"], label=f"{name} - Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 3, 2)
for name, data in results.items():
    plt.plot(data["test_accuracies"], label=f"{name} - Test Accuracy")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 3, 3)
training_times = [data["training_time"] for data in results.values()]
plt.bar(results.keys(), training_times)
plt.title("Total Training Time")
plt.ylabel("Time (seconds)")

plt.tight_layout()
plt.savefig("optimizer_comparison.png")
plt.show()

# Print final results
for name, data in results.items():
    print(f"\n{name} Results:")
    print(f"Final Test Accuracy: {data['test_accuracies'][-1]:.2f}%")
    print(f"Total Training Time: {data['training_time']:.2f} seconds")

# Test set: Average loss: 0.0809, Accuracy: 9764/10000 (97.64%)
