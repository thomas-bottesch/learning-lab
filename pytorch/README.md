# Pruning results

The results demonstrate that the initial model was overparameterized for the MNIST task. Structured pruning reduced the parameter count by 54.2%, with accuracy recovering to nearly the original level after fine-tuning. Applying Quantization-Aware Training (QAT) and converting to INT8 further reduced the persisted model size by 87.2%, with minimal impact on accuracy.

Results of pruning.py can be seen here:

vscode@TBO-Linux:/workspaces/learning-lab/pytorch/shrink_model$ python ./pruning.py

============================================================
STEP 1: Training initial float model
============================================================
Float Epoch 1/25, Train Loss: 0.8082, Val Loss: 0.1846, Val Accuracy: 94.42%
Float Epoch 2/25, Train Loss: 0.3602, Val Loss: 0.1465, Val Accuracy: 95.70%
Float Epoch 3/25, Train Loss: 0.2906, Val Loss: 0.1372, Val Accuracy: 95.51%
Float Epoch 4/25, Train Loss: 0.2553, Val Loss: 0.1009, Val Accuracy: 96.90%
Float Epoch 5/25, Train Loss: 0.2387, Val Loss: 0.0897, Val Accuracy: 97.20%
Float Epoch 6/25, Train Loss: 0.2266, Val Loss: 0.0968, Val Accuracy: 97.07%
Float Epoch 7/25, Train Loss: 0.2148, Val Loss: 0.0796, Val Accuracy: 97.58%
Float Epoch 8/25, Train Loss: 0.2092, Val Loss: 0.0746, Val Accuracy: 97.67%
Float Epoch 9/25, Train Loss: 0.2001, Val Loss: 0.0739, Val Accuracy: 97.88%
Float Epoch 10/25, Train Loss: 0.1935, Val Loss: 0.0741, Val Accuracy: 97.74%
Float Epoch 11/25, Train Loss: 0.1898, Val Loss: 0.0745, Val Accuracy: 97.97%
Float Epoch 12/25, Train Loss: 0.1812, Val Loss: 0.0683, Val Accuracy: 97.86%
Float Epoch 13/25, Train Loss: 0.1818, Val Loss: 0.0642, Val Accuracy: 98.03%
Float Epoch 14/25, Train Loss: 0.1741, Val Loss: 0.0595, Val Accuracy: 98.25%
Float Epoch 15/25, Train Loss: 0.1775, Val Loss: 0.0577, Val Accuracy: 98.26%
Float Epoch 16/25, Train Loss: 0.1718, Val Loss: 0.0626, Val Accuracy: 98.11%
Float Epoch 17/25, Train Loss: 0.1683, Val Loss: 0.0582, Val Accuracy: 98.28%
Float Epoch 18/25, Train Loss: 0.1688, Val Loss: 0.0603, Val Accuracy: 98.22%
Float Epoch 19/25, Train Loss: 0.1670, Val Loss: 0.0573, Val Accuracy: 98.18%
Float Epoch 20/25, Train Loss: 0.1645, Val Loss: 0.0552, Val Accuracy: 98.36%
Float Epoch 21/25, Train Loss: 0.1633, Val Loss: 0.0590, Val Accuracy: 98.36%
Float Epoch 22/25, Train Loss: 0.1633, Val Loss: 0.0580, Val Accuracy: 98.16%
Float Epoch 23/25, Train Loss: 0.1613, Val Loss: 0.0519, Val Accuracy: 98.44%
Float Epoch 24/25, Train Loss: 0.1648, Val Loss: 0.0576, Val Accuracy: 98.19%
Float Epoch 25/25, Train Loss: 0.1566, Val Loss: 0.0532, Val Accuracy: 98.32%

✓ Step 1 Complete - Float model accuracy: 98.26%

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

✓ Step 2 Complete - Pruned model (before fine-tuning) accuracy: 82.86%

============================================================
STEP 3: Fine-tuning pruned model
============================================================
Fine-tune Epoch 1/10, Train Loss: 0.3342, Val Loss: 0.0749, Val Accuracy: 97.75%
Fine-tune Epoch 2/10, Train Loss: 0.2446, Val Loss: 0.0705, Val Accuracy: 97.88%
Fine-tune Epoch 3/10, Train Loss: 0.2327, Val Loss: 0.0657, Val Accuracy: 98.05%
Fine-tune Epoch 4/10, Train Loss: 0.2210, Val Loss: 0.0636, Val Accuracy: 98.06%
Fine-tune Epoch 5/10, Train Loss: 0.2118, Val Loss: 0.0620, Val Accuracy: 98.17%
Fine-tune Epoch 6/10, Train Loss: 0.2121, Val Loss: 0.0605, Val Accuracy: 98.17%
Fine-tune Epoch 7/10, Train Loss: 0.2068, Val Loss: 0.0607, Val Accuracy: 98.14%
Fine-tune Epoch 8/10, Train Loss: 0.2022, Val Loss: 0.0590, Val Accuracy: 98.35%
Fine-tune Epoch 9/10, Train Loss: 0.1966, Val Loss: 0.0600, Val Accuracy: 98.32%
Fine-tune Epoch 10/10, Train Loss: 0.1931, Val Loss: 0.0572, Val Accuracy: 98.28%

✓ Step 3 Complete - Structured pruned model (after fine-tuning) accuracy: 98.22%

============================================================
STEP 4: Applying Quantization-Aware Training (QAT)
============================================================
Model prepared for QAT.
QAT Epoch 1/10, Train Loss: 0.1840, Val Loss: 0.0551, Val Accuracy: 98.32%
QAT Epoch 2/10, Train Loss: 0.1782, Val Loss: 0.0525, Val Accuracy: 98.47%
QAT Epoch 3/10, Train Loss: 0.1781, Val Loss: 0.0522, Val Accuracy: 98.47%
QAT Epoch 4/10, Train Loss: 0.1774, Val Loss: 0.0528, Val Accuracy: 98.49%
Observers frozen for stable quantization parameters.
QAT Epoch 5/10, Train Loss: 0.1779, Val Loss: 0.0520, Val Accuracy: 98.47%
QAT Epoch 6/10, Train Loss: 0.1782, Val Loss: 0.0520, Val Accuracy: 98.49%
QAT Epoch 7/10, Train Loss: 0.1773, Val Loss: 0.0515, Val Accuracy: 98.54%
QAT Epoch 8/10, Train Loss: 0.1719, Val Loss: 0.0519, Val Accuracy: 98.54%
QAT Epoch 9/10, Train Loss: 0.1712, Val Loss: 0.0511, Val Accuracy: 98.58%
QAT Epoch 10/10, Train Loss: 0.1705, Val Loss: 0.0505, Val Accuracy: 98.58%

✓ Step 4 Complete - QAT model (before conversion) accuracy: 98.52%

============================================================
STEP 5: Converting to quantized (INT8) model
============================================================
Model converted to INT8.

✓ Step 5 Complete - Quantized (INT8) model accuracy: 98.30%

============================================================
SUMMARY OF ALL STEPS
============================================================
Step 1 - Initial float model:                  98.26% (242,762 params, 948.3 KB)
Step 2 - Structured pruned (before fine-tune): 82.86% (111,146 params, 434.2 KB)
Step 3 - Structured pruned (after fine-tune):  98.22% (111,146 params, 434.2 KB)
Step 4 - QAT model (before conversion):        98.52%
Step 5 - Final quantized (INT8) model:         98.30% (persisted size: 121.6 KB)

============================================================
Total parameter reduction: 54.2%
Total size reduction: 87.2% (948.3 KB -> 121.6 KB)
============================================================