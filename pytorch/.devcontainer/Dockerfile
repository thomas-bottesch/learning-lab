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
    onnxruntime-gpu

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