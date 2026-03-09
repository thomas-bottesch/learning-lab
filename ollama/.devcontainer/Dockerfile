ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.2
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
FROM ${BASE_CUDA_DEV_CONTAINER}

ENV MY_USERNAME=vscode \
    MY_GROUPNAME=vscode \
    MY_USERID=1000 \
    MY_GROUPID=1000

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
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Install Ollama
# ------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set Ollama environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_ORIGINS=*

# Create Ollama models directory in user's home and set permissions
RUN mkdir -p /home/${MY_USERNAME}/.ollama/models \
    && chown -R ${MY_USERNAME}:${MY_GROUPNAME} /home/${MY_USERNAME}/.ollama

# Copy entrypoint script
COPY --chown=${MY_USERNAME}:${MY_GROUPNAME} entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to vscode user
USER ${MY_USERNAME}

# Set working directory
WORKDIR /workspace

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
