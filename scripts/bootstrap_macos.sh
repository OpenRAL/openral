#!/usr/bin/env bash
set -euo pipefail

# macOS 14+ bootstrap.
# Note: ROS 2 on macOS is best run via Docker (see Dockerfile.dev).
# This script sets up the Python/tooling side only.

if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew first: https://brew.sh" >&2
  exit 1
fi

brew install cmake ninja just llvm libusb python@3.12 jq

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ""
echo "==> macOS system bootstrap complete."
echo "==> ROS 2 on macOS is best run via Docker:"
echo "==>   docker compose -f docker-compose.dev.yml up -d openral-dev"
