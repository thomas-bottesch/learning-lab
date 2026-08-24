FROM ubuntu:26.04

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
    && useradd --gid ${MY_GROUPID} --create-home --uid ${MY_USERID} --shell /bin/bash ${MY_USERNAME}

# Install sudo, add vscode to sudo group, and allow passwordless sudo
RUN apt-get update && apt-get install -y sudo \
    && usermod -aG sudo ${MY_USERNAME} \
    && echo "${MY_USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${MY_USERNAME} \
    && chmod 0440 /etc/sudoers.d/${MY_USERNAME} \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ------------------------------
# Docker repository (for docker-ce-cli)
# ------------------------------
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    unzip \
    screen \
    curl \
    zstd \
    ca-certificates \
    locales \
    docker-ce-cli \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python

# ------------------------------
# Option B: venv OUTSIDE workspace
# ------------------------------
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create venv as root, then switch to MY_USER for pip installs to avoid chown
RUN python -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
    && chown -R ${MY_USERID}:${MY_GROUPID} ${VIRTUAL_ENV}

USER ${MY_USERNAME}
COPY requirements.txt /tmp/requirements.txt

# Make uv available for the vscode user
ENV PATH="/home/${MY_USERNAME}/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install -r /tmp/requirements.txt

RUN echo 'export HISTFILE=/home/vscode/.bash_history' >> ~/.bashrc \
    && echo 'shopt -s histappend' >> ~/.bashrc \
    && echo 'PROMPT_COMMAND="history -a; history -n; $PROMPT_COMMAND"' >> ~/.bashrc

RUN ${VIRTUAL_ENV}/bin/pip install -r /tmp/requirements.txt

RUN sudo rm /tmp/requirements.txt

# ------------------------------
# Do NOT set WORKDIR to /workspaces/*
# VS Code will mount the workspace itself
# ------------------------------
WORKDIR /workspace
