FROM docker.io/vllm/vllm-openai:nightly@sha256:00cc19f41e3427bbcf43941afcb3a8b5ae82c6b48bbab7922d885f83bd4d2dad

ENV MY_USERNAME=vscode \
    MY_GROUPNAME=vscode \
    MY_USERID=1000 \
    MY_GROUPID=1000

RUN if id ubuntu >/dev/null 2>&1; then \
    old_uid=$(id -u ubuntu) && \
    old_gid=$(id -g ubuntu) && \
    usermod -u 2000 ubuntu && \
    groupmod -g 2000 ubuntu && \
    find / -xdev -user $old_uid -exec chown -h 2000 {} \; 2>/dev/null || true && \
    find / -xdev -group $old_gid -exec chgrp -h 2000 {} \; 2>/dev/null || true; \
    fi

RUN groupadd -g ${MY_GROUPID} ${MY_GROUPNAME} \
    && useradd --gid ${MY_GROUPID} --create-home --uid ${MY_USERID} --shell /bin/bash ${MY_USERNAME} \
    && mkdir -p /workspace \
    && chown ${MY_USERNAME}:${MY_GROUPNAME} /workspace \
    && chmod 775 /workspace

RUN apt-get update && apt-get install -y sudo curl \
    && usermod -aG sudo ${MY_USERNAME} \
    && echo "${MY_USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${MY_USERNAME} \
    && chmod 0440 /etc/sudoers.d/${MY_USERNAME} \
    && rm -rf /var/lib/apt/lists/*

# Install Open WebUI
RUN pip install open-webui

COPY --chown=${MY_USERNAME}:${MY_GROUPNAME} entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
