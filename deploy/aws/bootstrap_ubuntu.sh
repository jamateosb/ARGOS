#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/argos/ARGOS}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

install_base_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update
    sudo apt-get install -y \
      git \
      curl \
      jq \
      ca-certificates \
      python3 \
      python3-venv \
      python3-pip \
      docker.io \
      docker-compose-plugin
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    local packages=()
    command -v git >/dev/null 2>&1 || packages+=(git)
    command -v jq >/dev/null 2>&1 || packages+=(jq)
    command -v curl >/dev/null 2>&1 || packages+=(curl)
    command -v python3 >/dev/null 2>&1 || packages+=(python3)
    python3 -m pip --version >/dev/null 2>&1 || packages+=(python3-pip)
    command -v docker >/dev/null 2>&1 || packages+=(docker)
    rpm -q ca-certificates >/dev/null 2>&1 || packages+=(ca-certificates)
    if ((${#packages[@]})); then
      sudo dnf install -y "${packages[@]}"
    fi
    if sudo dnf info docker-compose-plugin >/dev/null 2>&1; then
      sudo dnf install -y docker-compose-plugin
    fi
    return
  fi

  if command -v yum >/dev/null 2>&1; then
    local packages=()
    command -v git >/dev/null 2>&1 || packages+=(git)
    command -v jq >/dev/null 2>&1 || packages+=(jq)
    command -v curl >/dev/null 2>&1 || packages+=(curl)
    command -v python3 >/dev/null 2>&1 || packages+=(python3)
    python3 -m pip --version >/dev/null 2>&1 || packages+=(python3-pip)
    command -v docker >/dev/null 2>&1 || packages+=(docker)
    rpm -q ca-certificates >/dev/null 2>&1 || packages+=(ca-certificates)
    if ((${#packages[@]})); then
      sudo yum install -y "${packages[@]}"
    fi
    if sudo yum info docker-compose-plugin >/dev/null 2>&1; then
      sudo yum install -y docker-compose-plugin
    fi
    return
  fi

  echo "Unsupported package manager: expected apt-get, dnf, or yum" >&2
  exit 1
}

install_base_packages

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

cd "$PROJECT_DIR"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

sudo mkdir -p /etc/argos
echo "Bootstrap completed."
echo "Next: copy env templates from deploy/aws/env/*.env.example to /etc/."
echo "Optional validation: source .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q"
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin not detected. Use 'docker run' for NATS if needed."
fi
