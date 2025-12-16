## Overview

This example demonstrates the following workflow:

-  Build a machine learning model using PyTorch.
-  Convert the PyTorch model to the ONNX format.
-  Compile a new (small) ONNX runtime with only the operators required by the exported model.
-  Load the ONNX model using the new ONNX Runtime.
-  Perform inference with the loaded ONNX model using new ONNX Runtime.
-  Compare the inference results with those obtained from the original PyTorch model to insure correctness.

Follow this guide to explore the seamless integration of PyTorch and ONNX Runtime for efficient model deployment and inference.

## Prerequisites
This project is supposed to be loaded via vscode. Vscode will create a docker container a so-called "devcontainer" with all dependencies installed. Especially the following things are happening inside the Dockerfile:

The Dockerfile installs and configures the Python runtime, PyTorch, ONNX, and ONNX Runtime from the python repository. Then additionally it checks out the onnx runtime source code and installs it under /opt/onnxruntime. This folder will be used to compile a runtime with only the operators that are needed for the exported model.

## Execution

### Step 1: Build and Enter the Devcontainer
1. Open this project in Visual Studio Code.
2. If prompted, allow VS Code to reopen the project in a container. This will build the Docker image and start a container with all necessary dependencies installed.
3. If not prompted, you can manually trigger the container build by opening the Command Palette (Ctrl+Shift+P) and selecting "Remote-Containers: Reopen in Container".

### Step 2: Create the PyTorch Model and Export to ONNX

Run the following command in the terminal to execute the script that creates a PyTorch model and exports it to ONNX format:

```bash
cd pytorch_onnx
python train_model_and_convert_to_onnx.py
```

Files produced:
- model.onnx — The exported ONNX model.
- python_predictions.csv — CSV with columns: index,prediction,label,top_logit (inference results from PyTorch/ONNX Runtime Python run).

### Step 3: Build the minimal ONNX Runtime with Required Operators

The regular ONNX Runtime onnxruntime.so library has a size of about 13MB. To reduce the size we will build a minimal version of the library that only contains the operators that are needed for our exported model.

Run the following command in the terminal to build the minimal ONNX Runtime (attention this will take a few minutes!):

```bash
# <you are currently in folder pytorch_onnx>
python shrink_onnx_runtime_to_model_usage.py
```

What the shrink_onnx_runtime_to_model_usage.py script does:
- Analyzes the exported `model.onnx` to determine which ONNX operators are required (it uses tools/python/convert_onnx_models_to_ort.py to produce an operator config).
- Invokes the ONNX Runtime build (build.sh) with --minimal_build and --include_ops_by_config to compile a shared library containing only those operators.
- Strips unneeded symbols from the resulting libonnxruntime*.so to decrease size.

The script will generate the minimal ONNX Runtime shared library at:
`pytorch_onnx/custom_onnx_runtime/build/Release/libonnxruntime.so`

Additionally the script converted `model.onnx` to `pytorch_onnx/custom_onnx_runtime/model.ort`.
The newly created runtime is only able to run "ORT" models an not regular ONNX models.

Files produced:
- custom_onnx_runtime/build/Release/libonnxruntime.so — The minimal ONNX Runtime shared library.
- custom_onnx_runtime/build/* — Build artifacts from the ONNX Runtime build.
- custom_onnx_runtime/model.ort — The exported model in ORT format.
- custom_onnx_runtime/model.required_operators.config — The operator config used to build the minimal runtime.


### Step 4: Run Inference with ONNX Runtime in C

The repository includes a small C program (mnist_onnx_infer.c) and a Makefile to run inference over the MNIST test set using the (minimal) ONNX Runtime produced by the previos step.

This step requires the following items that were delivered by executing the previous steps:
- A built minimal ONNX Runtime and model produced by the shrink script:
  - Shared library: custom_onnx_runtime/build/Release/libonnxruntime.so
  - ORT model: custom_onnx_runtime/model.ort
  (these files are created by shrink_onnx_runtime_to_model_usage.py)
- MNIST raw test files at:
  - data/MNIST/raw/t10k-images-idx3-ubyte
  - data/MNIST/raw/t10k-labels-idx1-ubyte
  (these files are downloaded by train_model_and_convert_to_onnx.py)

```bash
# <you are currently in folder pytorch_onnx>

# Compile the c program:
make

# Run the program C program that uses the newly compiled ONNX Runtime to do inference on the MNIST testdata:
./mnist_onnx_infer
```

What the program does
- Loads custom_onnx_runtime/model.ort (hardcoded in the C program).
- Reads MNIST test images/labels from data/MNIST/raw. (hardcoded in the C program).
- Runs inference for each image and writes predictions to c_predictions.csv.

Files produced
- c_predictions.csv — CSV with columns: index,prediction,label,top_logit

### Step 5: Compare C Inference Results with PyTorch Inference Results

After you have produced both the Python and C inference outputs you can verify that the minimal ONNX Runtime produces identical results to the Python reference.

- Required files (produced by earlier steps):
  - python_predictions.csv — CSV with columns: index,prediction,label,top_logit (should contain PyTorch/ONNX Runtime Python run results)
  - c_predictions.csv — CSV with columns: index,prediction,label,top_logit (produced by the C program in Step 4)


```bash
# <you are currently in folder pytorch_onnx>

# Compile the c program:
make

# Run the program C program that uses the newly compiled ONNX Runtime to do inference on the MNIST testdata:
./mnist_onnx_infer
```

- If you don't yet have python_predictions.csv:
  - The project includes Python inference utilities (the training/export step often provides a script or helper to run inference and write python_predictions.csv). Run that script to produce python_predictions.csv. If unsure, search the folder for scripts that write "python_predictions.csv".

If all is well results should look like:
```bash
============================================================
Comparison Results:
============================================================
Total samples compared: 10000
Matching predictions (and logits within tol):   10000
Mismatching predictions/logits: 0
Match rate:             100.00%
```

