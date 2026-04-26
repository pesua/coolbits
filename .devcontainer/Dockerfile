FROM node:20-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    zsh \
    fzf \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install docker compose plugin from GitHub (architecture-aware)
RUN mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Add node user to docker group so it can talk to the mounted socket
RUN usermod -aG docker node

USER node

ENV PATH="/home/node/.local/bin:$PATH"

# Install Claude Code as node user so it lands in /home/node/.local/bin
RUN curl -fsSL https://claude.ai/install.sh | bash


WORKDIR /workspaces
