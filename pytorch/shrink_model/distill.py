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

# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.quantized.engine = "fbgemm"


# LeNet-5 style CNN for MNIST, handling normalization inside the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input: [B, 1, 28, 28], float32, range [0, 1]
        # Normalize to mean=0.1307, std=0.3081 (standard MNIST normalization)
        self.register_buffer("mean", torch.tensor(0.1307, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(0.3081, dtype=torch.float32))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28], float32, range [0, 1]
        # Normalize to match standard MNIST preprocessing
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StudentNet(nn.Module):
    def __init__(self, conv1_out=8, conv2_out=16, fc1_out=32):
        super(StudentNet, self).__init__()
        # Quantization stubs for QAT support
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv2_out * 7 * 7, fc1_out)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_out, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    step_name: str = "",
    observer_freeze_epoch: int = None,
    verbose: bool = True,
) -> nn.Module:
    model.to(device)
    """General training loop for DRY principle."""
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        if verbose:
            print(
                f"{step_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"
            )
        if observer_freeze_epoch is not None and epoch == observer_freeze_epoch:
            model.apply(torch.quantization.disable_observer)
            print("Observers frozen for stable quantization parameters.")
    return model


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
        inputs = inputs.to(device)
        labels = labels.to(device)
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss / len(loader), accuracy


def evaluate_model(
    model: nn.Module, loader: DataLoader, eval_device: torch.device = device
) -> float:
    """Evaluate model on a dataset and return accuracy."""
    model.to(eval_device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(eval_device)
            labels = labels.to(eval_device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def identify_neurons_to_keep_studentnet(model: nn.Module, amount: float = 0.5) -> dict:
    """Identify which output channels/neurons to keep based on L2 norm importance."""
    neurons_to_keep = {}

    for name, module in model.named_modules():
        # For Conv layers, prune output channels
        if isinstance(module, nn.Conv2d) and name in ["conv1", "conv2"]:
            # Compute L2 norm per output channel (dim=[1,2,3] for weights [out, in, h, w])
            importance = torch.norm(
                module.weight.data.view(module.out_channels, -1), p=2, dim=1
            )
            num_to_keep = int(module.out_channels * (1 - amount))
            _, indices = torch.topk(importance, num_to_keep, sorted=True)
            neurons_to_keep[name] = indices.sort()[0]
        # For Linear layers, prune output neurons (except final layer)
        elif isinstance(module, nn.Linear) and name == "fc1":
            importance = torch.norm(module.weight.data, p=2, dim=1)
            num_to_keep = int(module.out_features * (1 - amount))
            _, indices = torch.topk(importance, num_to_keep, sorted=True)
            neurons_to_keep[name] = indices.sort()[0]

    return neurons_to_keep


def create_pruned_studentnet(
    original_model: nn.Module, neurons_to_keep: dict
) -> nn.Module:
    """Create a new StudentNet with reduced layer sizes based on pruning decisions."""
    # Determine new layer sizes
    conv1_out = len(neurons_to_keep.get("conv1", torch.arange(8)))
    conv2_out = len(neurons_to_keep.get("conv2", torch.arange(16)))
    fc1_out = len(neurons_to_keep.get("fc1", torch.arange(32)))

    print(
        f"Creating pruned StudentNet: conv1(1->{conv1_out}), conv2({conv1_out}->{conv2_out}), "
        f"fc1({conv2_out}*7*7->{fc1_out}), fc2({fc1_out}->10)"
    )

    # Create new model with reduced sizes
    new_model = StudentNet(conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out)

    # Copy weights for kept channels/neurons
    with torch.no_grad():
        # conv1: Keep selected output channels
        if "conv1" in neurons_to_keep:
            indices = neurons_to_keep["conv1"]
            new_model.conv1.weight.data = original_model.conv1.weight.data[indices]
            new_model.conv1.bias.data = original_model.conv1.bias.data[indices]

        # conv2: Keep selected output channels AND corresponding input channels from conv1
        if "conv2" in neurons_to_keep:
            out_indices = neurons_to_keep["conv2"]
            in_indices = neurons_to_keep.get(
                "conv1", torch.arange(original_model.conv2.in_channels)
            )
            new_model.conv2.weight.data = original_model.conv2.weight.data[out_indices][
                :, in_indices
            ]
            new_model.conv2.bias.data = original_model.conv2.bias.data[out_indices]
        elif "conv1" in neurons_to_keep:
            # conv2 not pruned but conv1 was, adjust input dimension only
            in_indices = neurons_to_keep["conv1"]
            new_model.conv2.weight.data = original_model.conv2.weight.data[
                :, in_indices
            ]
            new_model.conv2.bias.data = original_model.conv2.bias.data.clone()

        # fc1: Keep selected output neurons AND adjust input based on conv2
        conv2_out_channels = len(
            neurons_to_keep.get(
                "conv2", torch.arange(original_model.conv2.out_channels)
            )
        )
        original_fc1_in = conv2_out_channels * 7 * 7

        if "fc1" in neurons_to_keep:
            out_indices = neurons_to_keep["fc1"]
            # Reshape to handle the flattened conv output
            if "conv2" in neurons_to_keep:
                # Need to select the right input features based on conv2 pruning
                in_indices = neurons_to_keep["conv2"]
                # Create a mask for the flattened features
                mask = torch.zeros(original_model.fc1.in_features, dtype=torch.bool)
                for i in in_indices:
                    mask[i * 49 : (i + 1) * 49] = True
                new_model.fc1.weight.data = original_model.fc1.weight.data[out_indices][
                    :, mask
                ]
            else:
                new_model.fc1.weight.data = original_model.fc1.weight.data[out_indices]
            new_model.fc1.bias.data = original_model.fc1.bias.data[out_indices]
        elif "conv2" in neurons_to_keep:
            # fc1 not pruned but conv2 was, adjust input dimension only
            in_indices = neurons_to_keep["conv2"]
            mask = torch.zeros(original_model.fc1.in_features, dtype=torch.bool)
            for i in in_indices:
                mask[i * 49 : (i + 1) * 49] = True
            new_model.fc1.weight.data = original_model.fc1.weight.data[:, mask]
            new_model.fc1.bias.data = original_model.fc1.bias.data.clone()

        # fc2: Only adjust input dimension based on fc1 pruning
        in_indices = neurons_to_keep.get(
            "fc1", torch.arange(original_model.fc2.in_features)
        )
        new_model.fc2.weight.data = original_model.fc2.weight.data[:, in_indices]
        new_model.fc2.bias.data = original_model.fc2.bias.data.clone()

    return new_model


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in the model."""
    param_count = sum(p.numel() for p in model.parameters() if p is not None)
    buffer_count = sum(b.numel() for b in model.buffers() if b is not None)
    return param_count + buffer_count


def clone_model_studentnet(model: nn.Module) -> nn.Module:
    """Clone a StudentNet model, preserving its architecture."""
    try:
        conv1_out = model.conv1.out_channels
        conv2_out = model.conv2.out_channels
        fc1_out = model.fc1.out_features
    except AttributeError:
        # Handle fused modules
        conv1_out = (
            model.conv1[0].out_channels if isinstance(model.conv1, nn.Sequential) else 8
        )
        conv2_out = (
            model.conv2[0].out_channels
            if isinstance(model.conv2, nn.Sequential)
            else 16
        )
        fc1_out = (
            model.fc1[0].out_features if isinstance(model.fc1, nn.Sequential) else 32
        )

    cloned = StudentNet(conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out)
    cloned.load_state_dict(model.state_dict(), strict=False)
    cloned.eval()
    return cloned


def prepare_qat_studentnet(float_model: nn.Module) -> nn.Module:
    """Prepare StudentNet for Quantization-Aware Training (QAT)."""
    # Fuse Conv+ReLU and Linear+ReLU for better QAT performance
    torch.quantization.fuse_modules(
        float_model,
        [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
        inplace=True,
    )

    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
        ),
    )

    float_model.qconfig = qconfig
    torch.quantization.prepare_qat(float_model, inplace=True)
    print("StudentNet prepared for QAT.")
    return float_model


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    student_logits, teacher_logits: [batch, num_classes]
    labels: [batch]
    T: temperature
    alpha: weight for distillation loss
    """
    soft_loss = nn.KLDivLoss(
        reduction="batchmean"
    )(  # KLDiv expects log-prob for input, prob for target
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1),
    ) * (
        T * T
    )
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def main() -> None:
    initial_epochs = 15

    # Data augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    # Preprocessing pipeline: PIL image -> Tensor -> Float32 -> Normalize to [-1, 1]
    # This preprocessing will need to be replicated in C code on the MCU
    # TODO: we must avoid converting from int8 -> float -> int8 in the MCU
    # Work is needed to avoid this!
    eval_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    full_train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    # For validation, use the same images but with eval_transform
    val_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=eval_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=eval_transform
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, _ = random_split(full_train_dataset, [train_size, val_size])
    _, val_subset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # StudentNet-specific DataLoaders for faster training
    student_train_loader = DataLoader(
        train_subset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )
    student_val_loader = DataLoader(
        val_subset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    # ====== STEP 1: Train initial float model ======
    float_model_path = f"/tmp/lenet5_model_{initial_epochs}"
    if os.path.exists(float_model_path):
        print(f"Step 1: Model already exists at {float_model_path}, skipping training.")
        model = SimpleNN()
        model.load_state_dict(torch.load(float_model_path, map_location=device))
        model.to(device)
    else:
        print("\n" + "=" * 60)
        print("STEP 1: Training initial float model")
        print("=" * 60)
        model = SimpleNN()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            initial_epochs,
            step_name="Float",
        )
        torch.save(model.state_dict(), float_model_path)
    test_acc = evaluate_model(model, test_loader)
    print(f"\n✓ Step 1 Complete - Float model accuracy: {test_acc * 100:.2f}%")

    # ====== STEP 2: Distill knowledge into StudentNet ======
    student_model_path = "/tmp/studentnet_distilled.pth"
    if os.path.exists(student_model_path):
        print(
            f"Step 2: Model already exists at {student_model_path}, skipping distillation."
        )
        student = StudentNet().to(device)
        student.load_state_dict(torch.load(student_model_path, map_location=device))
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Knowledge Distillation into StudentNet")
        print("=" * 60)

        # Distillation hyperparameters
        distill_epochs = 60
        temperature = 4.0
        alpha = 0.7  # weight for distillation loss vs. hard label loss

        # Load teacher (SimpleNN) and create student
        teacher = SimpleNN().to(device)
        teacher.load_state_dict(torch.load(float_model_path, map_location=device))
        teacher.eval()
        student = StudentNet().to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)

        for epoch in range(distill_epochs):
            student.train()
            running_loss = 0.0
            for inputs, labels in student_train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                student_logits = student(inputs)
                loss = distillation_loss(
                    student_logits, teacher_logits, labels, temperature, alpha
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            val_loss, val_acc = validate_model(
                student, student_val_loader, nn.CrossEntropyLoss()
            )
            print(
                f"Distill Epoch {epoch+1}/{distill_epochs}, Train Loss: {running_loss/len(student_train_loader):.4f}, Val Acc: {val_acc*100:.2f}%"
            )
        torch.save(student.state_dict(), student_model_path)
    test_acc = evaluate_model(student, test_loader)
    print(f"\n✓ Step 2 Complete - StudentNet distilled accuracy: {test_acc * 100:.2f}%")

    # ====== STEP 3: Apply structured pruning to StudentNet ======
    prune_amount = 0.5  # Fraction to prune
    finetune_epochs = 10
    pruned_student_path = f"/tmp/studentnet_pruned_{prune_amount}.pth"
    pruned_student_arch_path = f"/tmp/studentnet_pruned_{prune_amount}_arch.pt"

    if os.path.exists(pruned_student_path) and os.path.exists(pruned_student_arch_path):
        print(
            f"Step 3: Model already exists at {pruned_student_path}, skipping pruning."
        )
        arch = torch.load(pruned_student_arch_path)
        student = StudentNet(
            conv1_out=arch["conv1_out"],
            conv2_out=arch["conv2_out"],
            fc1_out=arch["fc1_out"],
        )
        student.load_state_dict(torch.load(pruned_student_path, map_location=device))
        student.to(device)
        pruned_student_before_finetune = clone_model_studentnet(student)
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Applying structured pruning to StudentNet")
        print("=" * 60)
        original_params = count_parameters(student)
        print(f"Original StudentNet parameters: {original_params:,}")

        neurons_to_keep = identify_neurons_to_keep_studentnet(
            student, amount=prune_amount
        )
        student = create_pruned_studentnet(student, neurons_to_keep)
        student.to(device)

        pruned_params = count_parameters(student)
        reduction = 100 * (1 - pruned_params / original_params)
        print(
            f"Pruned StudentNet parameters: {pruned_params:,} ({reduction:.1f}% reduction)"
        )

        pruned_student_before_finetune = clone_model_studentnet(student)
        torch.save(student.state_dict(), pruned_student_path)
        arch = {
            "conv1_out": student.conv1.out_channels,
            "conv2_out": student.conv2.out_channels,
            "fc1_out": student.fc1.out_features,
        }
        torch.save(arch, pruned_student_arch_path)

    test_acc = evaluate_model(pruned_student_before_finetune, test_loader)
    print(
        f"\n✓ Step 3 Complete - Pruned StudentNet (before fine-tuning) accuracy: {test_acc * 100:.2f}%"
    )

    # ====== STEP 4: Fine-tune pruned StudentNet ======
    finetuned_student_path = f"/tmp/studentnet_finetuned_{finetune_epochs}.pth"
    finetuned_student_arch_path = f"/tmp/studentnet_finetuned_{finetune_epochs}_arch.pt"

    if os.path.exists(finetuned_student_path) and os.path.exists(
        finetuned_student_arch_path
    ):
        print(
            f"Step 4: Model already exists at {finetuned_student_path}, skipping fine-tuning."
        )
        arch = torch.load(finetuned_student_arch_path)
        student = StudentNet(
            conv1_out=arch["conv1_out"],
            conv2_out=arch["conv2_out"],
            fc1_out=arch["fc1_out"],
        )
        student.load_state_dict(torch.load(finetuned_student_path, map_location=device))
        student.to(device)
        pruned_student_after_finetune = clone_model_studentnet(student)
    else:
        print("\n" + "=" * 60)
        print("STEP 4: Fine-tuning pruned StudentNet")
        print("=" * 60)
        optimizer = optim.Adam(student.parameters(), lr=0.0005)
        student.to(device)
        student = train_model(
            student,
            student_train_loader,
            student_val_loader,
            nn.CrossEntropyLoss(),
            optimizer,
            finetune_epochs,
            step_name="Fine-tune",
        )
        pruned_student_after_finetune = clone_model_studentnet(student)
        torch.save(student.state_dict(), finetuned_student_path)
        arch = {
            "conv1_out": student.conv1.out_channels,
            "conv2_out": student.conv2.out_channels,
            "fc1_out": student.fc1.out_features,
        }
        torch.save(arch, finetuned_student_arch_path)

    test_acc = evaluate_model(pruned_student_after_finetune, test_loader)
    print(
        f"\n✓ Step 4 Complete - Pruned StudentNet (after fine-tuning) accuracy: {test_acc * 100:.2f}%"
    )

    # ====== STEP 5: Apply Quantization-Aware Training (QAT) ======
    qat_epochs = 10
    qat_student_path = f"/tmp/studentnet_qat_{qat_epochs}.pth"
    qat_student_arch_path = f"/tmp/studentnet_qat_{qat_epochs}_arch.pt"

    if os.path.exists(qat_student_path) and os.path.exists(qat_student_arch_path):
        print(f"Step 5: Model already exists at {qat_student_path}, skipping QAT.")
        arch = torch.load(qat_student_arch_path)
        student = StudentNet(
            conv1_out=arch["conv1_out"],
            conv2_out=arch["conv2_out"],
            fc1_out=arch["fc1_out"],
        )
        student.load_state_dict(
            torch.load(qat_student_path, map_location=device), strict=False
        )
        student.to(device)
    else:
        print("\n" + "=" * 60)
        print("STEP 5: Applying Quantization-Aware Training (QAT)")
        print("=" * 60)
        student.train()
        student.to(device)
        student = prepare_qat_studentnet(student)
        optimizer = optim.Adam(student.parameters(), lr=0.0001)
        student = train_model(
            student,
            student_train_loader,
            student_val_loader,
            nn.CrossEntropyLoss(),
            optimizer,
            qat_epochs,
            step_name="QAT",
            observer_freeze_epoch=int(qat_epochs * 0.3),
        )
        torch.save(student.state_dict(), qat_student_path)
        arch = {
            "conv1_out": (
                student.conv1.out_channels
                if hasattr(student.conv1, "out_channels")
                else student.conv1[0].out_channels
            ),
            "conv2_out": (
                student.conv2.out_channels
                if hasattr(student.conv2, "out_channels")
                else student.conv2[0].out_channels
            ),
            "fc1_out": (
                student.fc1.out_features
                if hasattr(student.fc1, "out_features")
                else student.fc1[0].out_features
            ),
        }
        torch.save(arch, qat_student_arch_path)

    qat_student_before_convert_acc = evaluate_model(student, test_loader)
    print(
        f"\n✓ Step 5 Complete - QAT StudentNet (before conversion) accuracy: {qat_student_before_convert_acc * 100:.2f}%"
    )

    # ====== STEP 6: Convert to quantized (INT8) model and persist ======
    quantized_student_path = f"/tmp/studentnet_quantized_{qat_epochs}.pth"
    quantized_student_arch_path = f"/tmp/studentnet_quantized_{qat_epochs}_arch.pt"
    quantized_student_fullmodel_path = f"/tmp/studentnet_quantized_{qat_epochs}_full.pt"

    if (
        os.path.exists(quantized_student_path)
        and os.path.exists(quantized_student_arch_path)
        and os.path.exists(quantized_student_fullmodel_path)
    ):
        print(
            f"Step 6: Model already exists at {quantized_student_path}, skipping quantization."
        )
        student = torch.load(quantized_student_fullmodel_path, map_location="cpu")
    else:
        print("\n" + "=" * 60)
        print("STEP 6: Converting to quantized (INT8) StudentNet")
        print("=" * 60)
        student.to("cpu")
        torch.quantization.convert(student.eval(), inplace=True)
        print("StudentNet converted to INT8.")
        torch.save(student.state_dict(), quantized_student_path)
        torch.save(student, quantized_student_fullmodel_path)
        try:
            arch = {
                "conv1_out": student.conv1.out_channels,
                "conv2_out": student.conv2.out_channels,
                "fc1_out": student.fc1.out_features,
            }
        except AttributeError:
            # Handle quantized module structure
            arch = torch.load(qat_student_arch_path)
        torch.save(arch, quantized_student_arch_path)

    test_acc = evaluate_model(student, test_loader, eval_device="cpu")
    print(
        f"\n✓ Step 6 Complete - Quantized (INT8) StudentNet accuracy: {test_acc * 100:.2f}%"
    )

    # ====== SUMMARY OF ALL STEPS ======
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL STEPS")
    print("=" * 60)

    # Load original distilled student for comparison
    original_student = StudentNet().to(device)
    original_student.load_state_dict(
        torch.load(student_model_path, map_location=device)
    )

    original_params = count_parameters(original_student)
    pruned_params = count_parameters(pruned_student_before_finetune)
    finetuned_params = count_parameters(pruned_student_after_finetune)

    original_size_kb = original_params * 4 / 1024
    pruned_size_kb = pruned_params * 4 / 1024
    finetuned_size_kb = finetuned_params * 4 / 1024

    param_reduction = 100 * (1 - pruned_params / original_params)

    print(
        f"Step 2 - Distilled StudentNet:                    {evaluate_model(original_student, test_loader) * 100:.2f}% ({original_params:,} params, {original_size_kb:.1f} KB)"
    )
    print(
        f"Step 3 - Pruned StudentNet (before fine-tune):    {evaluate_model(pruned_student_before_finetune, test_loader) * 100:.2f}% ({pruned_params:,} params, {pruned_size_kb:.1f} KB)"
    )
    print(
        f"Step 4 - Pruned StudentNet (after fine-tune):     {evaluate_model(pruned_student_after_finetune, test_loader) * 100:.2f}% ({finetuned_params:,} params, {finetuned_size_kb:.1f} KB)"
    )
    print(
        f"Step 5 - QAT StudentNet (before conversion):      {qat_student_before_convert_acc * 100:.2f}%"
    )

    file_size_bytes = os.path.getsize(quantized_student_fullmodel_path)
    file_size_kb = file_size_bytes / 1024
    print(
        f"Step 6 - Final quantized (INT8) StudentNet:       {test_acc * 100:.2f}% (persisted size: {file_size_kb:.1f} KB)"
    )

    print("\n" + "=" * 60)
    print(f"Total parameter reduction: {param_reduction:.1f}%")
    size_reduction = 100 * (1 - file_size_kb / original_size_kb)
    print(
        f"Total size reduction: {size_reduction:.1f}% ({original_size_kb:.1f} KB -> {file_size_kb:.1f} KB)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
