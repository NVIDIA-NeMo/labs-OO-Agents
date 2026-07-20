#!/bin/bash
set -e

# Install Docker CE on Ubuntu 25.10 (Questing Quokka).
# Uses the 'noble' (24.04) channel since Docker doesn't have a 25.10 release yet.
# Run with: bash util/install_docker.sh

USER="${1:-$USER}"
echo "Installing Docker CE for user: $USER"

# Dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repo (pinned to noble — no questing release yet)
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

# Add user to docker group so sudo isn't needed
sudo usermod -aG docker "$USER"

echo
echo "Docker installed: $(docker --version)"
echo
echo "IMPORTANT: run 'newgrp docker' or log out/in for group change to take effect."
echo "Then verify with: docker run --rm hello-world"
