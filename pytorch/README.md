# Distill results

The results demonstrate that the initial TeacherNet model was overparameterized for the MNIST task. Knowledge distillation into a smaller StudentNet preserved most of the accuracy. Structured pruning further reduced the parameter count by 74.6%, with accuracy recovering to nearly the original level after fine-tuning with distillation. Applying Quantization-Aware Training (QAT) and converting to INT8 further reduced the persisted model size by 82.9%. Remaining parameters=6794.


Step 1: Model already exists at /tmp/distill_1_float_model_15, skipping training.

✓ Step 1 Complete - Float model accuracy: 99.46%

============================================================
STEP 2: Knowledge Distillation into StudentNet
============================================================
Distill Epoch 1/60, Train Loss: 5.2702, Val Acc: 94.88%
Distill Epoch 2/60, Train Loss: 1.8568, Val Acc: 96.18%
Distill Epoch 3/60, Train Loss: 1.4297, Val Acc: 96.89%
Distill Epoch 4/60, Train Loss: 1.1399, Val Acc: 97.32%
Distill Epoch 5/60, Train Loss: 0.9464, Val Acc: 97.49%
Distill Epoch 6/60, Train Loss: 0.8203, Val Acc: 97.86%
Distill Epoch 7/60, Train Loss: 0.7344, Val Acc: 97.86%
Distill Epoch 8/60, Train Loss: 0.6708, Val Acc: 98.10%
Distill Epoch 9/60, Train Loss: 0.6291, Val Acc: 98.31%
Distill Epoch 10/60, Train Loss: 0.5878, Val Acc: 98.35%
Distill Epoch 11/60, Train Loss: 0.5545, Val Acc: 98.22%
Distill Epoch 12/60, Train Loss: 0.5195, Val Acc: 98.58%
Distill Epoch 13/60, Train Loss: 0.5022, Val Acc: 98.34%
Distill Epoch 14/60, Train Loss: 0.4813, Val Acc: 98.32%
Distill Epoch 15/60, Train Loss: 0.4623, Val Acc: 98.51%
Distill Epoch 16/60, Train Loss: 0.4470, Val Acc: 98.61%
Distill Epoch 17/60, Train Loss: 0.4347, Val Acc: 98.63%
Distill Epoch 18/60, Train Loss: 0.4194, Val Acc: 98.77%
Distill Epoch 19/60, Train Loss: 0.4081, Val Acc: 98.58%
Distill Epoch 20/60, Train Loss: 0.4023, Val Acc: 98.71%
Distill Epoch 21/60, Train Loss: 0.3876, Val Acc: 98.77%
Distill Epoch 22/60, Train Loss: 0.3764, Val Acc: 98.67%
Distill Epoch 23/60, Train Loss: 0.3671, Val Acc: 98.87%
Distill Epoch 24/60, Train Loss: 0.3610, Val Acc: 98.75%
Distill Epoch 25/60, Train Loss: 0.3555, Val Acc: 98.74%
Distill Epoch 26/60, Train Loss: 0.3463, Val Acc: 98.73%
Distill Epoch 27/60, Train Loss: 0.3423, Val Acc: 98.83%
Distill Epoch 28/60, Train Loss: 0.3404, Val Acc: 98.85%
Distill Epoch 29/60, Train Loss: 0.3361, Val Acc: 98.86%
Distill Epoch 30/60, Train Loss: 0.3294, Val Acc: 98.92%
Distill Epoch 31/60, Train Loss: 0.3280, Val Acc: 98.86%
Distill Epoch 32/60, Train Loss: 0.3181, Val Acc: 98.92%
Distill Epoch 33/60, Train Loss: 0.3163, Val Acc: 98.92%
Distill Epoch 34/60, Train Loss: 0.3141, Val Acc: 98.78%
Distill Epoch 35/60, Train Loss: 0.3086, Val Acc: 98.78%
Distill Epoch 36/60, Train Loss: 0.3107, Val Acc: 98.95%
Distill Epoch 37/60, Train Loss: 0.3034, Val Acc: 98.84%
Distill Epoch 38/60, Train Loss: 0.3048, Val Acc: 98.91%
Distill Epoch 39/60, Train Loss: 0.3001, Val Acc: 98.92%
Distill Epoch 40/60, Train Loss: 0.2961, Val Acc: 98.83%
Distill Epoch 41/60, Train Loss: 0.2939, Val Acc: 99.01%
Distill Epoch 42/60, Train Loss: 0.2935, Val Acc: 98.94%
Distill Epoch 43/60, Train Loss: 0.2878, Val Acc: 98.93%
Distill Epoch 44/60, Train Loss: 0.2896, Val Acc: 98.92%
Distill Epoch 45/60, Train Loss: 0.2874, Val Acc: 99.03%
Distill Epoch 46/60, Train Loss: 0.2843, Val Acc: 98.94%
Distill Epoch 47/60, Train Loss: 0.2807, Val Acc: 99.05%
Distill Epoch 48/60, Train Loss: 0.2806, Val Acc: 98.92%
Distill Epoch 49/60, Train Loss: 0.2825, Val Acc: 98.94%
Distill Epoch 50/60, Train Loss: 0.2819, Val Acc: 98.92%
Distill Epoch 51/60, Train Loss: 0.2782, Val Acc: 98.91%
Distill Epoch 52/60, Train Loss: 0.2778, Val Acc: 99.02%
Distill Epoch 53/60, Train Loss: 0.2771, Val Acc: 98.94%
Distill Epoch 54/60, Train Loss: 0.2758, Val Acc: 99.02%
Distill Epoch 55/60, Train Loss: 0.2740, Val Acc: 98.98%
Distill Epoch 56/60, Train Loss: 0.2731, Val Acc: 98.98%
Distill Epoch 57/60, Train Loss: 0.2704, Val Acc: 99.00%
Distill Epoch 58/60, Train Loss: 0.2713, Val Acc: 99.08%
Distill Epoch 59/60, Train Loss: 0.2691, Val Acc: 99.02%
Distill Epoch 60/60, Train Loss: 0.2698, Val Acc: 99.07%

✓ Step 2 Complete - StudentNet distilled accuracy: 99.16%

============================================================
STEP 3: Applying structured pruning to StudentNet
============================================================
Original StudentNet parameters: 26,698
Creating pruned StudentNet: conv1(1->4), conv2(4->8), fc1(8*7*7->16), fc2(16->10)
Pruned StudentNet parameters: 6,794 (74.6% reduction)

✓ Step 3 Complete - Pruned StudentNet (before fine-tuning) accuracy: 17.37%

============================================================
STEP 4: Fine-tuning pruned StudentNet with teacher distillation
============================================================
Fine-tune (distillation) Epoch 1/40, Train Loss: 3.5770, Val Loss: 0.1062, Val Accuracy: 96.63%
Fine-tune (distillation) Epoch 2/40, Train Loss: 1.1622, Val Loss: 0.0793, Val Accuracy: 97.56%
Fine-tune (distillation) Epoch 3/40, Train Loss: 0.9171, Val Loss: 0.0683, Val Accuracy: 97.95%
Fine-tune (distillation) Epoch 4/40, Train Loss: 0.8079, Val Loss: 0.0632, Val Accuracy: 98.08%
Fine-tune (distillation) Epoch 5/40, Train Loss: 0.7418, Val Loss: 0.0609, Val Accuracy: 98.17%
Fine-tune (distillation) Epoch 6/40, Train Loss: 0.7010, Val Loss: 0.0603, Val Accuracy: 98.18%
Fine-tune (distillation) Epoch 7/40, Train Loss: 0.6632, Val Loss: 0.0577, Val Accuracy: 98.27%
Fine-tune (distillation) Epoch 8/40, Train Loss: 0.6414, Val Loss: 0.0545, Val Accuracy: 98.32%
Fine-tune (distillation) Epoch 9/40, Train Loss: 0.6202, Val Loss: 0.0557, Val Accuracy: 98.32%
Fine-tune (distillation) Epoch 10/40, Train Loss: 0.6123, Val Loss: 0.0565, Val Accuracy: 98.30%
Fine-tune (distillation) Epoch 11/40, Train Loss: 0.5946, Val Loss: 0.0560, Val Accuracy: 98.34%
Fine-tune (distillation) Epoch 12/40, Train Loss: 0.5898, Val Loss: 0.0555, Val Accuracy: 98.42%
Fine-tune (distillation) Epoch 13/40, Train Loss: 0.5759, Val Loss: 0.0536, Val Accuracy: 98.47%
Fine-tune (distillation) Epoch 14/40, Train Loss: 0.5760, Val Loss: 0.0541, Val Accuracy: 98.45%
Fine-tune (distillation) Epoch 15/40, Train Loss: 0.5685, Val Loss: 0.0522, Val Accuracy: 98.49%
Fine-tune (distillation) Epoch 16/40, Train Loss: 0.5580, Val Loss: 0.0523, Val Accuracy: 98.44%
Fine-tune (distillation) Epoch 17/40, Train Loss: 0.5566, Val Loss: 0.0528, Val Accuracy: 98.42%
Fine-tune (distillation) Epoch 18/40, Train Loss: 0.5545, Val Loss: 0.0519, Val Accuracy: 98.42%
Fine-tune (distillation) Epoch 19/40, Train Loss: 0.5474, Val Loss: 0.0517, Val Accuracy: 98.43%
Fine-tune (distillation) Epoch 20/40, Train Loss: 0.5458, Val Loss: 0.0536, Val Accuracy: 98.36%
Fine-tune (distillation) Epoch 21/40, Train Loss: 0.5415, Val Loss: 0.0502, Val Accuracy: 98.46%
Fine-tune (distillation) Epoch 22/40, Train Loss: 0.5380, Val Loss: 0.0527, Val Accuracy: 98.40%
Fine-tune (distillation) Epoch 23/40, Train Loss: 0.5339, Val Loss: 0.0508, Val Accuracy: 98.46%
Fine-tune (distillation) Epoch 24/40, Train Loss: 0.5395, Val Loss: 0.0509, Val Accuracy: 98.46%
Fine-tune (distillation) Epoch 25/40, Train Loss: 0.5334, Val Loss: 0.0493, Val Accuracy: 98.52%
Fine-tune (distillation) Epoch 26/40, Train Loss: 0.5250, Val Loss: 0.0507, Val Accuracy: 98.47%
Fine-tune (distillation) Epoch 27/40, Train Loss: 0.5233, Val Loss: 0.0496, Val Accuracy: 98.48%
Fine-tune (distillation) Epoch 28/40, Train Loss: 0.5328, Val Loss: 0.0498, Val Accuracy: 98.49%
Fine-tune (distillation) Epoch 29/40, Train Loss: 0.5242, Val Loss: 0.0501, Val Accuracy: 98.51%
Fine-tune (distillation) Epoch 30/40, Train Loss: 0.5269, Val Loss: 0.0488, Val Accuracy: 98.49%
Fine-tune (distillation) Epoch 31/40, Train Loss: 0.5221, Val Loss: 0.0491, Val Accuracy: 98.47%
Fine-tune (distillation) Epoch 32/40, Train Loss: 0.5195, Val Loss: 0.0497, Val Accuracy: 98.40%
Fine-tune (distillation) Epoch 33/40, Train Loss: 0.5227, Val Loss: 0.0489, Val Accuracy: 98.50%
Fine-tune (distillation) Epoch 34/40, Train Loss: 0.5178, Val Loss: 0.0485, Val Accuracy: 98.46%
Fine-tune (distillation) Epoch 35/40, Train Loss: 0.5174, Val Loss: 0.0508, Val Accuracy: 98.40%
Fine-tune (distillation) Epoch 36/40, Train Loss: 0.5169, Val Loss: 0.0480, Val Accuracy: 98.49%
Fine-tune (distillation) Epoch 37/40, Train Loss: 0.5203, Val Loss: 0.0497, Val Accuracy: 98.41%
Fine-tune (distillation) Epoch 38/40, Train Loss: 0.5096, Val Loss: 0.0497, Val Accuracy: 98.38%
Fine-tune (distillation) Epoch 39/40, Train Loss: 0.5105, Val Loss: 0.0485, Val Accuracy: 98.40%
Fine-tune (distillation) Epoch 40/40, Train Loss: 0.5108, Val Loss: 0.0477, Val Accuracy: 98.53%

✓ Step 4 Complete - Pruned StudentNet (after fine-tuning) accuracy: 98.71%

============================================================
STEP 5: Applying Quantization-Aware Training (QAT) with distillation
============================================================
StudentNet prepared for QAT.
QAT (distillation) Epoch 1/10, Train Loss: 0.4989, Val Loss: 0.0485, Val Accuracy: 98.48%
QAT (distillation) Epoch 2/10, Train Loss: 0.5050, Val Loss: 0.0486, Val Accuracy: 98.47%
QAT (distillation) Epoch 3/10, Train Loss: 0.5003, Val Loss: 0.0479, Val Accuracy: 98.52%
Observers frozen for stable quantization parameters.
QAT (distillation) Epoch 4/10, Train Loss: 0.5017, Val Loss: 0.0484, Val Accuracy: 98.49%
QAT (distillation) Epoch 5/10, Train Loss: 0.5009, Val Loss: 0.0492, Val Accuracy: 98.49%
QAT (distillation) Epoch 6/10, Train Loss: 0.4957, Val Loss: 0.0486, Val Accuracy: 98.49%
QAT (distillation) Epoch 7/10, Train Loss: 0.4995, Val Loss: 0.0482, Val Accuracy: 98.50%
QAT (distillation) Epoch 8/10, Train Loss: 0.4994, Val Loss: 0.0487, Val Accuracy: 98.51%
QAT (distillation) Epoch 9/10, Train Loss: 0.4980, Val Loss: 0.0489, Val Accuracy: 98.48%
QAT (distillation) Epoch 10/10, Train Loss: 0.4958, Val Loss: 0.0479, Val Accuracy: 98.50%

✓ Step 5 Complete - QAT StudentNet (before conversion) accuracy: 98.79%

============================================================
STEP 6: Converting to quantized (INT8) StudentNet
============================================================
StudentNet converted to INT8.

✓ Step 6 Complete - Quantized (INT8) StudentNet accuracy: 98.25%

============================================================
SUMMARY OF ALL STEPS
============================================================
Step 2 - Distilled StudentNet:                    99.16% (26,698 params, 104.3 KB)
Step 3 - Pruned StudentNet (before fine-tune):    17.37% (6,794 params, 26.5 KB)
Step 4 - Pruned StudentNet (after fine-tune):     98.71% (6,794 params, 26.5 KB)
Step 5 - QAT StudentNet (before conversion):      98.79%
Step 6 - Final quantized (INT8) StudentNet:       98.25% (persisted size: 17.8 KB)

============================================================
Total parameter reduction: 74.6%
Total size reduction: 82.9% (104.3 KB -> 17.8 KB)
============================================================

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