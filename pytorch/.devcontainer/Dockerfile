FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV MY_USERNAME=vscode \
    MY_GROUPNAME=vscode \
    MY_USERID=1000 \
    MY_GROUPID=1000

# Reassign existing 'ubuntu' user to avoid UID conflict
# (only if it exists)
RUN if id ubuntu >/dev/null 2>&1; then \
        old_uid=$(id -u ubuntu) && \
        old_gid=$(id -g ubuntu) && \
        echo "Changing 'ubuntu' from UID:$old_uid to UID:2000" && \
        usermod -u 2000 ubuntu && \
        groupmod -g 2000 ubuntu && \
        find / -xdev -user $old_uid -exec chown -h 2000 {} \; 2>/dev/null || true && \
        find / -xdev -group $old_gid -exec chgrp -h 2000 {} \; 2>/dev/null || true; \
    fi

RUN set -x \
    && groupadd -g ${MY_GROUPID} ${MY_GROUPNAME} \
    && useradd --gid ${MY_GROUPID} --create-home --uid ${MY_USERID} --shell /bin/bash ${MY_USERNAME}

# Install sudo, add vscode to sudo group, and allow passwordless sudo for vscode
RUN apt-get update && apt-get install -y sudo \
    && usermod -aG sudo ${MY_USERNAME} \
    && echo "${MY_USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${MY_USERNAME} \
    && chmod 0440 /etc/sudoers.d/${MY_USERNAME} \
    && rm -rf /var/lib/apt/lists/*

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
    valgrind \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip & setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# ------------------------------
# PyTorch (CUDA 12.4). Not using 13.x as 12.4 is more widely supported
# ------------------------------
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------
# ONNX + ONNX Runtime (GPU)
# ------------------------------
RUN pip install \
    onnx \
    onnxruntime-gpu \
    onnxruntime-tools

# ------------------------------
# Quality-of-life Python tools
# ------------------------------
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    ipykernel \
    tensorboard

# ------------------------------
# Default working directory
# ------------------------------
WORKDIR /workspace

# ------------------------------
# Verify CUDA at build time to see if devcontainer setup was correct
# ------------------------------
RUN python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# ------------------------------
# Set up history configuration
# ------------------------------
RUN echo "HISTSIZE=20000" >> /home/vscode/.bashrc && \
    echo "HISTFILESIZE=40000" >> /home/vscode/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /home/vscode/.bashrc && \
    chown vscode:vscode /home/vscode/.bashrc

# ------------------------------
# ONNX Runtime source for custom minimal builds
# ------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y \
        cmake \
        libprotobuf-dev \
        libeigen3-dev \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Clone ONNX Runtime source code
# The replacement of the has is due to a bug in the ONNX Runtime v1.17.0 deps.txt file:
# The issue is discussed here: https://github.com/microsoft/onnxruntime/pull/18287
# -- SHA1 hash of
# /workspaces/learning-lab/pytorch/pytorch_onnx/custom_onnx_runtime/build/Release/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-e7248b26a1ed53fa030c5c459f7ea095dfd276ac.zip
# does not match expected value
#   expected: 'be8be39fdbc6e60e94fa7870b280707069b5b81a'
#    actual: '32b145f525a8308d7ab1c09388b2e288312d8eba'
WORKDIR /opt
RUN git clone --recursive --branch v1.17.0 https://github.com/microsoft/onnxruntime.git && \
    if [ -f /opt/onnxruntime/cmake/deps.txt ]; then \
    sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/g' /opt/onnxruntime/cmake/deps.txt; \
    fi && \
    chown -R vscode:vscode /opt/onnxruntime

# Set environment variable for ONNX Runtime source
ENV ONNXRUNTIME_SOURCE=/opt/onnxruntime

# Verify ONNX Runtime source is available
RUN python - <<EOF
import os
ort_source = os.environ.get('ONNXRUNTIME_SOURCE')
print(f"ONNX Runtime source location: {ort_source}")
print(f"Source exists: {os.path.exists(ort_source)}")
if os.path.exists(ort_source):
    print(f"Build script exists: {os.path.exists(os.path.join(ort_source, 'build.sh'))}")
EOF

# Return to default workspace
WORKDIR /workspace