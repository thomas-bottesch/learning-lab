# Build and run the MNIST ONNX Runtime C inference program

CC = gcc
#CFLAGS = -O2 -std=c99 -I/opt/onnxruntime/include  
CFLAGS = -std=c99 -I/opt/onnxruntime/include  -g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls
LDFLAGS = -Lcustom_onnx_runtime/build/Release -lonnxruntime
RPATH = -Wl,-rpath,'$$ORIGIN/custom_onnx_runtime/build/Release'

TARGET = mnist_onnx_infer

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): mnist_onnx_infer.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(RPATH)

run: build
	# Provide model and raw MNIST test files. Adjust paths if needed.
	LD_LIBRARY_PATH=custom_onnx_runtime/build/Release:$$LD_LIBRARY_PATH ./$(TARGET) model.onnx data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte

clean:
	rm -f $(TARGET)
