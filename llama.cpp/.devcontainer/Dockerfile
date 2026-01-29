ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.8.0
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# For GTX 1080=61 and for RTX 5090 GPUs=120

FROM ${BASE_CUDA_DEV_CONTAINER}

ARG CMAKE_CUDA_ARCHITECTURES="61"

ENV MY_USERNAME=vscode \
    MY_GROUPNAME=vscode \
    MY_USERID=1000 \
    MY_GROUPID=1000 \
    HF_HOME=/workspace/git_repo/.cache/huggingface

# llama-server configuration
ENV LLAMA_SERVER_HOST=0.0.0.0 \
    LLAMA_SERVER_PORT=8080

# Reassign existing 'ubuntu' user to avoid UID conflict
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
    && useradd --gid ${MY_GROUPID} --create-home --uid ${MY_USERID} --shell /bin/bash ${MY_USERNAME} \
    && mkdir -p /workspace \
    && chown ${MY_USERNAME}:${MY_GROUPNAME} /workspace \
    && chmod 775 /workspace

# Install sudo, add vscode to sudo group, and allow passwordless sudo
RUN apt-get update && apt-get install -y sudo \
    && usermod -aG sudo ${MY_USERNAME} \
    && echo "${MY_USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${MY_USERNAME} \
    && chmod 0440 /etc/sudoers.d/${MY_USERNAME} \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    unzip \
    screen \
    curl \
    zstd \
    ca-certificates \
    build-essential \
    cmake \
    libssl-dev \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Make `python` and `python3` point to `python3.12`
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# ------------------------------
# Option B: venv OUTSIDE workspace
# ------------------------------
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create venv as root using Python 3.12, then switch to MY_USER for uv installs to avoid chown
RUN python3.12 -m venv ${VIRTUAL_ENV} \
    && chown -R ${MY_USERID}:${MY_GROUPID} ${VIRTUAL_ENV}

USER ${MY_USERNAME}

# Clone and build llama.cpp
# LD_LIBRARY_PATH is set to find the CUDA stubs during build
# else cmake may fail to find libcudart.so
RUN mkdir -p /home/${MY_USERNAME}/repos \
    && cd /home/${MY_USERNAME}/repos \
    && git clone https://github.com/ggml-org/llama.cpp \
    && cd llama.cpp \
    && git checkout 88d23ad \
    && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j7

COPY requirements.txt /tmp/requirements.txt
# Make uv available for the vscode user
ENV PATH="/home/${MY_USERNAME}/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install -r /tmp/requirements.txt

RUN echo 'export HISTFILE=/home/vscode/.bash_history' >> ~/.bashrc \
    && echo 'shopt -s histappend' >> ~/.bashrc \
    && echo 'PROMPT_COMMAND="history -a; history -n; $PROMPT_COMMAND"' >> ~/.bashrc

USER root

RUN rm /tmp/requirements.txt

# ------------------------------
# WORKDIR is automatically handled by devcontainer.json's workspaceFolder setting
# No need to set it here - VS Code will cd to the correct location
# ------------------------------
