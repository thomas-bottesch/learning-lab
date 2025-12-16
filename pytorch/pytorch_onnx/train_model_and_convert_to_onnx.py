# This file is represents the following steps:
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
import onnxruntime as ort
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from typing import Tuple


# Step 1: Model Creation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # integrate preprocessing into the model so ONNX handles it:
        # - input is expected as uint8 image [0,255] (shape [N,1,28,28])
        # - convert to float, scale to [0,1], then normalize to [-1,1] (same as (x/255 - 0.5)/0.5)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            x = x / 255.0
            x = (x - 0.5) / 0.5
        return self.model(x)


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
    # use PILToTensor so dataset yields uint8 tensors (0-255) and model does normalization
    transform = transforms.PILToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: "
            f"{val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%"
        )

    return model


# Step 2: ONNX Export
def export_to_onnx(model: nn.Module, filepath: str = "model.onnx") -> None:
    # export with a uint8 dummy input so the preprocessing (Cast/Div/Sub) is included in ONNX
    dummy_input = torch.randint(0, 256, (1, 1, 28, 28), dtype=torch.uint8)
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Model exported to {filepath}")


# Step 3 & 4: Inference and Comparison
def run_inference_with_onnx_and_compare(
    onnx_model_path: str, test_loader: DataLoader, pytorch_model: nn.Module
) -> None:
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: None}
    all_onnx_preds = []
    all_onnx_top_logits = []
    all_pytorch_preds = []
    all_pytorch_top_logits = []
    all_labels = []

    pytorch_model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # PyTorch outputs (logits)
            pt_outputs = pytorch_model(inputs)
            pt_out_np = pt_outputs.detach().numpy()
            pytorch_preds = np.argmax(pt_out_np, axis=1)
            pytorch_top_logits = pt_out_np[np.arange(len(pytorch_preds)), pytorch_preds]
            all_pytorch_preds.extend(pytorch_preds.tolist())
            all_pytorch_top_logits.extend(pytorch_top_logits.tolist())

            # ONNX Runtime outputs (logits)
            ort_inputs[ort_session.get_inputs()[0].name] = inputs.numpy()
            ort_outs = ort_session.run(None, ort_inputs)
            onnx_out = ort_outs[0]
            onnx_preds = np.argmax(onnx_out, axis=1)
            onnx_top_logits = onnx_out[np.arange(len(onnx_preds)), onnx_preds]
            all_onnx_preds.extend(onnx_preds.tolist())
            all_onnx_top_logits.extend(onnx_top_logits.tolist())

            all_labels.extend(labels.numpy().tolist())

    # Write predictions to CSV (index,prediction,label,top_logit)
    csv_path = "python_predictions.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "prediction", "label", "top_logit"])
        for i, (pred, label, top_logit) in enumerate(
            zip(all_onnx_preds, all_labels, all_onnx_top_logits)
        ):
            writer.writerow([i, int(pred), int(label), float(f"{top_logit:.6f}")])
    print(f"Predictions written to {csv_path}")

    identical = np.array_equal(all_pytorch_preds, all_onnx_preds)
    print(f"Are PyTorch and ONNX predictions identical? {identical}")

    if not identical:
        for i, (p, o, l, p_logit, o_logit) in enumerate(
            zip(
                all_pytorch_preds,
                all_onnx_preds,
                all_labels,
                all_pytorch_top_logits,
                all_onnx_top_logits,
            )
        ):
            if p != o:
                print(
                    f"Mismatch at index {i}: PyTorch={p} (logit={p_logit:.6f}), "
                    f"ONNX={o} (logit={o_logit:.6f}), Label={l}"
                )
        raise ValueError("Predictions from PyTorch and ONNX models do not match!")

    pytorch_accuracy = accuracy_score(all_labels, all_pytorch_preds)
    onnx_accuracy = accuracy_score(all_labels, all_onnx_preds)
    print(f"PyTorch Model Accuracy: {pytorch_accuracy * 100:.2f}%")
    print(f"ONNX Model Accuracy: {onnx_accuracy * 100:.2f}%")


def main() -> None:
    model = train_model()
    onnx_model_path = "model.onnx"
    export_to_onnx(model, onnx_model_path)

    # use the same PILToTensor transform for test dataset (no extra normalization step required)
    transform = transforms.PILToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    run_inference_with_onnx_and_compare(onnx_model_path, test_loader, model)
    print(
        "Model 'model.onnx': ONNX model inference matches PyTorch model inference successfully."
    )


if __name__ == "__main__":
    main()
