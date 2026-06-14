#!/usr/bin/env bash
set -euo pipefail

# Idempotent Ollama bootstrap for the OpenRAL local reasoner baseline.
#
# Installs the `ollama` binary (Linux: official install.sh; macOS:
# `brew install`), starts the daemon if it isn't already running, and
# pulls the default baseline model (qwen3:8b — strong tool-use, ~5 GB).
#
# After this script, the three env vars below point the reasoner at the
# local endpoint:
#
#   export OPENRAL_REASONER_LLM_PROVIDER=openai-compatible
#   export OPENRAL_REASONER_LLM_MODEL=qwen3:8b
#   export OPENRAL_REASONER_LLM_BASE_URL=http://localhost:11434/v1
#
# Flags:
#   --no-pull           Skip the `ollama pull` step (binary install only).
#   --model <tag>       Pull a different model tag (default: qwen3:8b).

MODEL="qwen3:8b"
DO_PULL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pull) DO_PULL=0; shift ;;
    --model)   MODEL="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bootstrap_ollama.sh [--no-pull] [--model <tag>]

Installs Ollama (if missing), starts the daemon, and pulls the local
reasoner baseline model (qwen3:8b by default).

  --no-pull         Skip the `ollama pull` step (binary install only).
  --model <tag>     Pull a different model tag (default: qwen3:8b).

Afterwards, export these to wire the reasoner at the local endpoint:
  export OPENRAL_REASONER_LLM_PROVIDER=openai-compatible
  export OPENRAL_REASONER_LLM_MODEL=qwen3:8b
  export OPENRAL_REASONER_LLM_BASE_URL=http://localhost:11434/v1
EOF
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

OS="$(uname -s)"

# 1. Install the ollama binary if missing.
if command -v ollama >/dev/null 2>&1; then
  echo "==> ollama already installed: $(command -v ollama)"
else
  case "${OS}" in
    Linux)
      echo "==> installing ollama via official installer"
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        echo "==> installing ollama via brew"
        brew install ollama
      else
        echo "Homebrew not found. Install from https://ollama.com/download/mac" >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported OS: ${OS}. Install ollama manually: https://ollama.com" >&2
      exit 1
      ;;
  esac
fi

# 2. Start the daemon if the OpenAI-compatible endpoint isn't reachable.
#    `ollama serve` blocks; on Linux it's typically managed via systemd
#    by the installer, on macOS it's started by the menu-bar app. We
#    only background a `serve` if no listener answers on 11434.
probe_port() {
  python3 - "$@" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

if probe_port localhost 11434; then
  echo "==> ollama daemon already serving on localhost:11434"
else
  echo "==> starting ollama serve in the background"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  # Wait up to ~5 s for the listener to come up before giving up.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    if probe_port localhost 11434; then break; fi
  done
  if ! probe_port localhost 11434; then
    echo "ollama serve failed to bind on localhost:11434 — see /tmp/ollama.log" >&2
    exit 1
  fi
fi

# 3. Pull the baseline model (idempotent — ollama re-uses cached layers).
if [[ "${DO_PULL}" -eq 1 ]]; then
  echo "==> pulling ${MODEL} (will reuse layers if already cached)"
  ollama pull "${MODEL}"
fi

cat <<EOF

==> Local reasoner baseline ready. Export these to wire it up:

    export OPENRAL_REASONER_LLM_PROVIDER=openai-compatible
    export OPENRAL_REASONER_LLM_MODEL=${MODEL}
    export OPENRAL_REASONER_LLM_BASE_URL=http://localhost:11434/v1

Then \`openral doctor\` should report a green "Reasoner LLM" + "Ollama" row.
EOF
