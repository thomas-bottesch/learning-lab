# This file demonstrates structured pruning and quantization-aware training (QAT) for MCU deployment.
# Pipeline steps:
# 1. Train a full-precision (float32) model
# 2. Apply structured pruning (remove entire output neurons in Linear layers)
# 3. Fine-tune the pruned model to recover accuracy
# 4. Apply QAT (Quantization-Aware Training) to prepare for INT8 quantization
# 5. Convert the model to quantized INT8 and persist it
# Each step is evaluated on test data. Model size and accuracy are reported at each stage.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from typing import Tuple

torch.backends.quantized.engine = "fbgemm"


# Model Creation
class SimpleNN(nn.Module):
    def __init__(self, fc1_out=256, fc2_out=128, fc3_out=64):
        super(SimpleNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, fc1_out)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_out, fc3_out)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(fc3_out, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Expects preprocessed float32 input normalized to [-1, 1]
        # QuantStub/DeQuantStub are used for QAT and quantization conversion
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


def identify_neurons_to_keep(model: nn.Module, amount: float = 0.5) -> dict:
    """Identify which output neurons to keep in each Linear layer based on L2 norm importance (structured pruning)."""
    neurons_to_keep = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "fc4":  # Don't prune output layer
            # Compute L2 norm per output neuron (dim=1 for weights)
            importance = torch.norm(module.weight.data, p=2, dim=1)
            num_to_keep = int(module.out_features * (1 - amount))
            # Get indices of most important neurons
            _, indices = torch.topk(importance, num_to_keep, sorted=True)
            neurons_to_keep[name] = indices.sort()[0]  # Sort for consistent ordering

    return neurons_to_keep


def create_pruned_model(original_model: nn.Module, neurons_to_keep: dict) -> nn.Module:
    """Create a new model with reduced layer sizes based on structured pruning decisions."""
    # Determine new layer sizes
    fc1_out = len(neurons_to_keep.get("fc1", torch.arange(256)))
    fc2_out = len(neurons_to_keep.get("fc2", torch.arange(128)))
    fc3_out = len(neurons_to_keep.get("fc3", torch.arange(64)))

    print(
        f"Creating pruned model: fc1({28*28}->{fc1_out}), fc2({fc1_out}->{fc2_out}), fc3({fc2_out}->{fc3_out}), fc4({fc3_out}->10)"
    )

    # Create new model with reduced sizes
    new_model = SimpleNN(fc1_out=fc1_out, fc2_out=fc2_out, fc3_out=fc3_out)

    # Copy weights for kept neurons
    with torch.no_grad():
        # fc1: Keep selected output neurons
        if "fc1" in neurons_to_keep:
            indices = neurons_to_keep["fc1"]
            new_model.fc1.weight.data = original_model.fc1.weight.data[indices]
            new_model.fc1.bias.data = original_model.fc1.bias.data[indices]

        # fc2: Keep selected output neurons AND corresponding input neurons from fc1
        if "fc2" in neurons_to_keep:
            out_indices = neurons_to_keep["fc2"]
            in_indices = neurons_to_keep.get(
                "fc1", torch.arange(original_model.fc2.in_features)
            )
            new_model.fc2.weight.data = original_model.fc2.weight.data[out_indices][
                :, in_indices
            ]
            new_model.fc2.bias.data = original_model.fc2.bias.data[out_indices]
        elif "fc1" in neurons_to_keep:
            # fc2 not pruned but fc1 was, adjust input dimension only
            in_indices = neurons_to_keep["fc1"]
            new_model.fc2.weight.data = original_model.fc2.weight.data[:, in_indices]
            new_model.fc2.bias.data = original_model.fc2.bias.data.clone()

        # fc3: Keep selected output neurons AND corresponding input neurons from fc2
        if "fc3" in neurons_to_keep:
            out_indices = neurons_to_keep["fc3"]
            in_indices = neurons_to_keep.get(
                "fc2", torch.arange(original_model.fc3.in_features)
            )
            new_model.fc3.weight.data = original_model.fc3.weight.data[out_indices][
                :, in_indices
            ]
            new_model.fc3.bias.data = original_model.fc3.bias.data[out_indices]
        elif "fc2" in neurons_to_keep:
            # fc3 not pruned but fc2 was, adjust input dimension only
            in_indices = neurons_to_keep["fc2"]
            new_model.fc3.weight.data = original_model.fc3.weight.data[:, in_indices]
            new_model.fc3.bias.data = original_model.fc3.bias.data.clone()

        # fc4: Only adjust input dimension based on fc3 pruning (never prune output - need all 10 classes)
        in_indices = neurons_to_keep.get(
            "fc3", torch.arange(original_model.fc4.in_features)
        )
        new_model.fc4.weight.data = original_model.fc4.weight.data[:, in_indices]
        new_model.fc4.bias.data = original_model.fc4.bias.data.clone()

    return new_model


# Pruning is physical: layers are actually reduced in size, so no masks or reparametrization are needed.


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in the model, including quantized weights/buffers if present (for reporting only)."""
    param_count = sum(p.numel() for p in model.parameters() if p is not None)
    buffer_count = sum(b.numel() for b in model.buffers() if b is not None)

    return param_count + buffer_count


def get_model_size_info(model: nn.Module) -> dict:
    """Get detailed model size information for each Linear layer (for reporting only)."""
    total_params = 0
    layer_info = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            params = module.weight.numel() + module.bias.numel()
            total_params += params
            layer_info[name] = {
                "shape": f"{module.in_features}x{module.out_features}",
                "params": params,
            }

    return {"total_params": total_params, "layers": layer_info}


def clone_model(model: nn.Module) -> nn.Module:
    """Clone a model, preserving its current architecture (including pruned sizes)."""
    # Detect layer sizes from the original model
    fc1_out = model.fc1.out_features
    fc2_out = model.fc2.out_features
    fc3_out = model.fc3.out_features

    # Create new model with same architecture
    cloned = SimpleNN(fc1_out=fc1_out, fc2_out=fc2_out, fc3_out=fc3_out)

    # Load the state dict
    cloned.load_state_dict(model.state_dict())
    cloned.eval()
    return cloned


def prepare_qat_model(float_model: nn.Module) -> nn.Module:
    """Prepare model for Quantization-Aware Training (QAT)."""

    # TODO: to optimize MCU performance we could fuse the Linear-RELU layers here

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
    """Train the model for one epoch."""
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
    """Evaluate model on validation set and return loss and accuracy."""
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


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate model on a dataset and return accuracy."""
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


def main() -> None:
    # Configuration
    # Adjust these values as needed for your experiment
    initial_epochs = 25  # Initial training epochs
    finetune_epochs = 10  # Fine-tuning epochs after pruning
    qat_epochs = 10  # QAT epochs
    prune_amount = 0.5  # Fraction of output neurons to prune in each Linear layer

    # Prepare data
    # Preprocessing pipeline: PIL image -> Tensor -> Float -> Normalize to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ====== STEP 1: Train initial float model ======
    print("\n" + "=" * 60)
    print("STEP 1: Training initial float model")
    print("=" * 60)

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(initial_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{initial_epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
        )

    # Evaluate after Step 1
    float_model_initial = clone_model(model)
    test_acc = evaluate_model(float_model_initial, test_loader)
    print(f"\n✓ Step 1 Complete - Float model accuracy: {test_acc * 100:.2f}%")

    # ====== STEP 2: Apply structured pruning (physical model shrinking) ======
    print("\n" + "=" * 60)
    print("STEP 2: Applying structured pruning (physical model shrinking)")
    print("=" * 60)

    original_params = count_parameters(model)
    print(f"Original model parameters: {original_params:,}")

    # Identify which neurons to keep
    neurons_to_keep = identify_neurons_to_keep(model, amount=prune_amount)

    # Create physically smaller model
    model = create_pruned_model(model, neurons_to_keep)
    pruned_params = count_parameters(model)
    reduction = 100 * (1 - pruned_params / original_params)
    print(f"Pruned model parameters: {pruned_params:,} ({reduction:.1f}% reduction)")

    size_info = get_model_size_info(model)
    print("Layer sizes:")
    for layer_name, info in size_info["layers"].items():
        print(f"  {layer_name}: {info['shape']} ({info['params']:,} params)")

    # Evaluate after Step 2 (immediately after pruning, before fine-tuning)
    pruned_model_before_finetune = clone_model(model)
    test_acc = evaluate_model(pruned_model_before_finetune, test_loader)
    print(
        f"\n✓ Step 2 Complete - Pruned model (before fine-tuning) accuracy: {test_acc * 100:.2f}%"
    )

    # ====== STEP 3: Fine-tune pruned model ======
    print("\n" + "=" * 60)Pl
    print("STEP 3: Fine-tuning pruned model")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate

    for epoch in range(finetune_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{finetune_epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
        )

    # Evaluate after Step 3
    pruned_model_after_finetune = clone_model(model)
    test_acc = evaluate_model(pruned_model_after_finetune, test_loader)
    print(
        f"\n✓ Step 3 Complete - Structured pruned model (after fine-tuning) accuracy: {test_acc * 100:.2f}%"
    )

    # ====== STEP 4: Apply Quantization-Aware Training (QAT) ======
    print("\n" + "=" * 60)
    print("STEP 4: Applying Quantization-Aware Training (QAT)")
    print("=" * 60)

    model.train()  # QAT requires model to be in training mode
    model = prepare_qat_model(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for QAT

    for epoch in range(qat_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{qat_epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
        )

        # Freeze observers at 30% through training for stability
        if epoch == int(qat_epochs * 0.3):
            model.apply(torch.quantization.disable_observer)
            print("Observers frozen for stable quantization parameters.")

    # Evaluate before quantization conversion
    qat_model_before_convert_acc = evaluate_model(model, test_loader)
    print(
        f"\n✓ Step 4 Complete - QAT model (before conversion) accuracy: {qat_model_before_convert_acc * 100:.2f}%"
    )

    # ====== STEP 5: Convert to quantized (INT8) model and persist ======
    print("\n" + "=" * 60)
    print("STEP 5: Converting to quantized (INT8) model")
    print("=" * 60)

    torch.quantization.convert(model.eval(), inplace=True)
    print("Model converted to INT8.")

    # Evaluate after Step 5
    test_acc = evaluate_model(model, test_loader)
    print(
        f"\n✓ Step 5 Complete - Quantized (INT8) model accuracy: {test_acc * 100:.2f}%"
    )

    # ====== SUMMARY OF ALL STEPS ======
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL STEPS")
    print("=" * 60)

    # Count parameters for each model stage
    float_params = count_parameters(float_model_initial)
    pruned_params = count_parameters(pruned_model_before_finetune)
    finetuned_params = count_parameters(pruned_model_after_finetune)

    # Estimate model sizes (FP32 = 4 bytes, INT8 = 1 byte)
    float_size_kb = float_params * 4 / 1024
    pruned_size_kb = pruned_params * 4 / 1024
    finetuned_size_kb = finetuned_params * 4 / 1024

    param_reduction = 100 * (1 - pruned_params / float_params)

    print(
        f"Step 1 - Initial float model:                  {evaluate_model(float_model_initial, test_loader) * 100:.2f}% ({float_params:,} params, {float_size_kb:.1f} KB)"
    )
    print(
        f"Step 2 - Structured pruned (before fine-tune): {evaluate_model(pruned_model_before_finetune, test_loader) * 100:.2f}% ({pruned_params:,} params, {pruned_size_kb:.1f} KB)"
    )
    print(
        f"Step 3 - Structured pruned (after fine-tune):  {evaluate_model(pruned_model_after_finetune, test_loader) * 100:.2f}% ({finetuned_params:,} params, {finetuned_size_kb:.1f} KB)"
    )
    print(
        f"Step 4 - QAT model (before conversion):        {qat_model_before_convert_acc * 100:.2f}%"
    )
    # Report persisted file size for quantized model

    quantized_model_path = "/tmp/mnist_pruned_quantized_model.pth"
    torch.save(model.state_dict(), quantized_model_path)
    file_size_bytes = os.path.getsize(quantized_model_path)
    file_size_kb = file_size_bytes / 1024
    print(
        f"Step 5 - Final quantized (INT8) model:         {test_acc * 100:.2f}% (persisted size: {file_size_kb:.1f} KB)"
    )
    print("\n" + "=" * 60)
    print(f"Total parameter reduction: {param_reduction:.1f}%")
    size_reduction = 100 * (1 - file_size_kb / float_size_kb)
    print(
        f"Total size reduction: {size_reduction:.1f}% ({float_size_kb:.1f} KB -> {file_size_kb:.1f} KB)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
