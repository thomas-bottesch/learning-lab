ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.2
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
FROM ${BASE_CUDA_DEV_CONTAINER}

ENV MY_USERNAME=vscode \
    MY_GROUPNAME=vscode \
    MY_USERID=1000 \
    MY_GROUPID=1000 \
    HF_HOME=/workspace/git_repo/.cache/huggingface

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
    git-lfs \
    wget \
    unzip \
    screen \
    curl \
    zstd \
    vim \
    ca-certificates \
    build-essential \
    cmake \
    libssl-dev \
    libgomp1 \
    docker.io \
    python3.12-venv \
    yq \
    && rm -rf /var/lib/apt/lists/*


# Install the kubectl & k9s for Kubernetes cluster management and debugging, kustomize, helm, and terraform
ENV KUBECTL_VERSION="v1.31.0"
ENV K9S_VERSION="v0.50.18"
ENV KUSTOMIZE_VERSION="v5.8.1"
ENV HELM_VERSION="v4.0.2"
ENV TERRAFORM_VERSION="1.14.5"
ENV KUBECTL_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
ENV K9S_URL="https://github.com/derailed/k9s/releases/download/${K9S_VERSION}/k9s_Linux_amd64.tar.gz"
ENV KUSTOMIZE_URL="https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize/${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_linux_amd64.tar.gz"
ENV HELM_SCRIPT_URL="https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4"
RUN curl -sSL -o /usr/local/bin/kubectl ${KUBECTL_URL} \
    && chmod +x /usr/local/bin/kubectl \
    && wget -O /tmp/k9s_Linux_amd64.tar.gz ${K9S_URL} \
    && tar xzf /tmp/k9s_Linux_amd64.tar.gz -C /usr/local/bin k9s \
    && rm /tmp/k9s_Linux_amd64.tar.gz \
    # Install kustomize
    && curl -Lo /tmp/kustomize_linux_amd64.tar.gz ${KUSTOMIZE_URL} \
    && tar -xzf /tmp/kustomize_linux_amd64.tar.gz -C /tmp \
    && mv /tmp/kustomize /usr/local/bin/ \
    && rm /tmp/kustomize_linux_amd64.tar.gz \
    # Install helm
    && curl -fsSL -o /tmp/get_helm.sh ${HELM_SCRIPT_URL} \
    && chmod +x /tmp/get_helm.sh \
    && /tmp/get_helm.sh --version ${HELM_VERSION} \
    && rm /tmp/get_helm.sh \
    # Install terraform
    && curl -fsSL https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip -o terraform.zip \
    && unzip terraform.zip \
    && mv terraform /usr/local/bin/terraform \
    && chmod +x /usr/local/bin/terraform \
    && rm terraform.zip

# Make `python` and `python3` point to `python3.12` (Ubuntu 24.04 has Python 3.12 as the default, but the symlinks may not be set up)
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# ------------------------------
# venv OUTSIDE workspace
# ------------------------------
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create venv as root using Python 3.12, then switch to MY_USER for uv installs to avoid chown
RUN python3.12 -m venv ${VIRTUAL_ENV} \
    && chown -R ${MY_USERID}:${MY_GROUPID} ${VIRTUAL_ENV}

# Install ORAS 1.3.0 (OCI Registry As Storage)
RUN curl -LO https://github.com/oras-project/oras/releases/download/v1.3.0/oras_1.3.0_linux_amd64.tar.gz && \
    mkdir -p /tmp/oras-install && \
    tar -zxf oras_1.3.0_linux_amd64.tar.gz -C /tmp/oras-install && \
    mv /tmp/oras-install/oras /usr/local/bin/oras && \
    chmod +x /usr/local/bin/oras && \
    rm -rf oras_1.3.0_linux_amd64.tar.gz /tmp/oras-install && \
    oras version

USER ${MY_USERNAME}

COPY requirements.txt /tmp/requirements.txt
# Make uv available for the vscode user
ENV PATH="/home/${MY_USERNAME}/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install -r /tmp/requirements.txt

RUN echo 'export HISTFILE=/home/vscode/.bash_history' >> ~/.bashrc \
    && echo 'shopt -s histappend' >> ~/.bashrc \
    && echo 'PROMPT_COMMAND="history -a; history -n; $PROMPT_COMMAND"' >> ~/.bashrc

# K9s settings
ENV XDG_HOME=/home/${MY_USERNAME}/.config
ENV XDG_CONFIG_HOME=/home/${MY_USERNAME}/.config
# Make k9s config dir and .kube directory.
# This is needed so that when mapping files into there from host, permissions are correct
RUN mkdir -p /home/${MY_USERNAME}/.config/k9s \
    && mkdir -p /home/${MY_USERNAME}/.local/share/k9s/clusters/default/default/

# Activate the venv in .bashrc so that the prompt correctly appears
RUN echo 'if [ -f "/opt/venv/bin/activate" ]; then' >> /home/${MY_USERNAME}/.bashrc \
    && echo '    source /opt/venv/bin/activate' >> /home/${MY_USERNAME}/.bashrc \
    && echo 'fi' >> /home/${MY_USERNAME}/.bashrc

USER root

RUN rm /tmp/requirements.txt

# ------------------------------
# WORKDIR is automatically handled by devcontainer.json's workspaceFolder setting
# No need to set it here - VS Code will cd to the correct location
# ------------------------------