# Pruning results

The results suggest that the initial model architecture was overparameterized for the MNIST task, as structured pruning led to a significant reduction in model size (54.2% fewer parameters) while maintaining and even improving accuracy after fine-tuning. The application of Quantization-Aware Training (QAT) further enhanced the model's efficiency by converting it to a quantized INT8 format, resulting in an 87.2% reduction in persisted size without a substantial loss in accuracy.

Results of pruning.py can be seen here:

============================================================
STEP 1: Training initial float model
============================================================
Epoch 1/25, Train Loss: 0.4297, Val Loss: 0.2695, Val Accuracy: 91.98%
Epoch 2/25, Train Loss: 0.1834, Val Loss: 0.1535, Val Accuracy: 95.65%
Epoch 3/25, Train Loss: 0.1316, Val Loss: 0.1433, Val Accuracy: 95.64%
Epoch 4/25, Train Loss: 0.1110, Val Loss: 0.1464, Val Accuracy: 95.43%
Epoch 5/25, Train Loss: 0.0903, Val Loss: 0.1079, Val Accuracy: 96.88%
Epoch 6/25, Train Loss: 0.0795, Val Loss: 0.0948, Val Accuracy: 97.17%
Epoch 7/25, Train Loss: 0.0666, Val Loss: 0.0991, Val Accuracy: 97.04%
Epoch 8/25, Train Loss: 0.0636, Val Loss: 0.1009, Val Accuracy: 97.02%
Epoch 9/25, Train Loss: 0.0573, Val Loss: 0.1091, Val Accuracy: 96.90%
Epoch 10/25, Train Loss: 0.0506, Val Loss: 0.0988, Val Accuracy: 97.24%
Epoch 11/25, Train Loss: 0.0459, Val Loss: 0.1031, Val Accuracy: 97.04%
Epoch 12/25, Train Loss: 0.0452, Val Loss: 0.1074, Val Accuracy: 97.01%
Epoch 13/25, Train Loss: 0.0447, Val Loss: 0.1063, Val Accuracy: 97.23%
Epoch 14/25, Train Loss: 0.0363, Val Loss: 0.0918, Val Accuracy: 97.47%
Epoch 15/25, Train Loss: 0.0323, Val Loss: 0.1027, Val Accuracy: 97.48%
Epoch 16/25, Train Loss: 0.0340, Val Loss: 0.1101, Val Accuracy: 97.18%
Epoch 17/25, Train Loss: 0.0313, Val Loss: 0.1192, Val Accuracy: 97.23%
Epoch 18/25, Train Loss: 0.0284, Val Loss: 0.1298, Val Accuracy: 96.90%
Epoch 19/25, Train Loss: 0.0289, Val Loss: 0.1111, Val Accuracy: 97.31%
Epoch 20/25, Train Loss: 0.0276, Val Loss: 0.1051, Val Accuracy: 97.41%
Epoch 21/25, Train Loss: 0.0240, Val Loss: 0.1163, Val Accuracy: 97.38%
Epoch 22/25, Train Loss: 0.0221, Val Loss: 0.1379, Val Accuracy: 96.95%
Epoch 23/25, Train Loss: 0.0251, Val Loss: 0.1121, Val Accuracy: 97.50%
Epoch 24/25, Train Loss: 0.0273, Val Loss: 0.1522, Val Accuracy: 96.49%
Epoch 25/25, Train Loss: 0.0193, Val Loss: 0.1227, Val Accuracy: 97.35%

✓ Step 1 Complete - Float model accuracy: 97.50%

============================================================
STEP 2: Applying structured pruning (physical model shrinking)
============================================================
Original model parameters: 242,762
Creating pruned model: fc1(784->128), fc2(128->64), fc3(64->32), fc4(32->10)
Pruned model parameters: 111,146 (54.2% reduction)
Layer sizes:
  fc1: 784x128 (100,480 params)
  fc2: 128x64 (8,256 params)
  fc3: 64x32 (2,080 params)
  fc4: 32x10 (330 params)

✓ Step 2 Complete - Pruned model (before fine-tuning) accuracy: 84.37%

============================================================
STEP 3: Fine-tuning pruned model
============================================================
Epoch 1/10, Train Loss: 0.0261, Val Loss: 0.0903, Val Accuracy: 97.88%
Epoch 2/10, Train Loss: 0.0110, Val Loss: 0.0900, Val Accuracy: 98.02%
Epoch 3/10, Train Loss: 0.0083, Val Loss: 0.0924, Val Accuracy: 98.02%
Epoch 4/10, Train Loss: 0.0063, Val Loss: 0.1008, Val Accuracy: 98.01%
Epoch 5/10, Train Loss: 0.0071, Val Loss: 0.1063, Val Accuracy: 97.80%
Epoch 6/10, Train Loss: 0.0052, Val Loss: 0.1021, Val Accuracy: 98.15%
Epoch 7/10, Train Loss: 0.0089, Val Loss: 0.1067, Val Accuracy: 97.86%
Epoch 8/10, Train Loss: 0.0029, Val Loss: 0.1208, Val Accuracy: 97.82%
Epoch 9/10, Train Loss: 0.0089, Val Loss: 0.1339, Val Accuracy: 97.70%
Epoch 10/10, Train Loss: 0.0040, Val Loss: 0.1082, Val Accuracy: 97.89%

✓ Step 3 Complete - Structured pruned model (after fine-tuning) accuracy: 97.98%

============================================================
STEP 4: Applying Quantization-Aware Training (QAT)
============================================================
/usr/local/lib/python3.10/dist-packages/torch/ao/quantization/observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Model prepared for QAT.
Epoch 1/10, Train Loss: 0.0008, Val Loss: 0.1060, Val Accuracy: 98.20%
Epoch 2/10, Train Loss: 0.0004, Val Loss: 0.1121, Val Accuracy: 98.17%
Epoch 3/10, Train Loss: 0.0002, Val Loss: 0.1133, Val Accuracy: 98.19%
Epoch 4/10, Train Loss: 0.0003, Val Loss: 0.1161, Val Accuracy: 98.17%
Observers frozen for stable quantization parameters.
Epoch 5/10, Train Loss: 0.0001, Val Loss: 0.1164, Val Accuracy: 98.22%
Epoch 6/10, Train Loss: 0.0001, Val Loss: 0.1243, Val Accuracy: 98.16%
Epoch 7/10, Train Loss: 0.0001, Val Loss: 0.1239, Val Accuracy: 98.19%
Epoch 8/10, Train Loss: 0.0004, Val Loss: 0.1269, Val Accuracy: 98.13%
Epoch 9/10, Train Loss: 0.0001, Val Loss: 0.1265, Val Accuracy: 98.17%
Epoch 10/10, Train Loss: 0.0000, Val Loss: 0.1304, Val Accuracy: 98.17%

✓ Step 4 Complete - QAT model (before conversion) accuracy: 98.13%

============================================================
STEP 5: Converting to quantized (INT8) model
============================================================
Model converted to INT8.

✓ Step 5 Complete - Quantized (INT8) model accuracy: 98.11%

============================================================
SUMMARY OF ALL STEPS
============================================================
Step 1 - Initial float model:                  97.50% (242,762 params, 948.3 KB)
Step 2 - Structured pruned (before fine-tune): 84.37% (111,146 params, 434.2 KB)
Step 3 - Structured pruned (after fine-tune):  97.98% (111,146 params, 434.2 KB)
Step 4 - QAT model (before conversion):        98.13%
Step 5 - Final quantized (INT8) model:         98.11% (persisted size: 121.6 KB)

============================================================
Total parameter reduction: 54.2%
Total size reduction: 87.2% (948.3 KB -> 121.6 KB)
============================================================