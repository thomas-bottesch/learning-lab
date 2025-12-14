# This file is supposed to do the following steps:
# 1. **Model Creation**: Build a machine learning model using PyTorch.
#    - For simplicity, we can use the mnist dataset with a simple feedforward neural network.
# 2. **ONNX Export**: Convert the PyTorch model to the ONNX format.
# 3. **Model Loading**: Load the ONNX model using ONNX Runtime.
# 4. **Inference Execution**: Perform inference with the loaded ONNX model using ONNX Runtime.
#    - We predict all samples with the PyTorch model and the ONNX model and compare the results.
#      The goal is to be sure that torch and onnx deliver the same results!


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os

# Step 1: Model Creation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Split dataset into training and validation sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(2):  # Train for 2 epochs
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy * 100:.2f}%")
        model.train()
    
    return model

# Step 2: ONNX Export
def export_to_onnx(model, filepath="model.onnx"):
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input size
    torch.onnx.export(model, dummy_input, filepath, input_names=['input'], output_names=['output'], opset_version=11)
    print(f"Model exported to {filepath}")

# Step 3: Model Loading and Step 4: Inference Execution
def run_inference_with_onnx(onnx_model_path, test_loader):
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: None}
    all_preds = []
    all_labels = []
    
    for inputs, labels in test_loader:
        ort_inputs[ort_session.get_inputs()[0].name] = inputs.numpy()
        ort_outs = ort_session.run(None, ort_inputs)
        preds = np.argmax(ort_outs[0], axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"ONNX Model Accuracy: {accuracy * 100:.2f}%")

def main():
    # Train PyTorch model
    model = train_model()
    
    # Export to ONNX
    onnx_model_path = "model.onnx"
    export_to_onnx(model, onnx_model_path)
    
    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Run inference with ONNX model
    run_inference_with_onnx(onnx_model_path, test_loader)

if __name__ == "__main__":
    main()
