# This file is used to learn Quantization-Aware Training (QAT) on a mnist model.
# 1. Train mnist float model
# 2. Quantize the model post training
# 3. Compare results
#
# Results with 25 epochs
# QAT Quantized (int8) model accuracy:   97.38%

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import csv
import os
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from typing import Tuple

torch.backends.quantized.engine = "fbgemm"


# Step 1: Model Creation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # integrate preprocessing into the model so ONNX handles it:
        # - input is expected as uint8 image [0,255] (shape [N,1,28,28])
        # - convert to float, scale to [0,1], then normalize to [-1,1] (same as (x/255 - 0.5)/0.5)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            x = x / 255.0
            x = (x - 0.5) / 0.5
        x = self.quant(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.dequant(x)
        return x


def prepare_qat_model(float_model: nn.Module) -> nn.Module:
    # Set QAT qconfig
    float_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(float_model, inplace=True)
    print("Model prepared for QAT.")
    return float_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss / len(loader), accuracy


def train_model() -> nn.Module:

    epochs = 25

    model_path = f"/tmp/mnist_qat_model_{epochs}.pth"
    transform = transforms.PILToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    if os.path.exists(model_path):
        print(f"Loading QAT model from {model_path}")
        model = SimpleNN()
        model.load_state_dict(torch.load(model_path))
        return model, train_loader

    model = SimpleNN()
    model = prepare_qat_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: "
            f"{val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%"
        )
        # Optionally, after a few epochs, freeze observers and BN stats
        if epoch == int(epochs * 0.75):
            model.apply(torch.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Convert to quantized model after QAT
    torch.quantization.convert(model.eval(), inplace=True)
    print("QAT model converted to quantized (INT8).")

    torch.save(model.state_dict(), model_path)
    print(f"Saved QAT model to {model_path}")
    return model, train_loader


# Evaluate a model on a DataLoader and return accuracy
def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# Compare float and quantized models on test data
def compare_models_on_test(
    float_model: nn.Module, quantized_model: nn.Module, test_loader: DataLoader
) -> None:
    float_acc = evaluate_model(float_model, test_loader)
    quant_acc = evaluate_model(quantized_model, test_loader)
    print(f"Original (float) model accuracy:   {float_acc * 100:.2f}%")
    print(f"Quantized (int8) model accuracy:   {quant_acc * 100:.2f}%")


def main() -> None:
    model, train_loader = train_model()
    # Prepare test loader for evaluation only
    transform = transforms.PILToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate quantized model on test data
    quant_acc = evaluate_model(model, test_loader)
    print(f"QAT Quantized (int8) model accuracy:   {quant_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
