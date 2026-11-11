#!/usr/bin/env bash
set -euo pipefail

# Idempotent Ollama bootstrap for the OpenRAL local reasoner baseline.
#
# Installs the `ollama` binary (Linux: upstream's install.sh at a pinned
# release tag, run only after its sha256 matches the pin below; macOS:
# `brew install`), starts the daemon if it isn't already running, and
# pulls the default baseline model (qwen3:8b — strong tool-use, ~5 GB).
#
# Model-first contract (ADR-0088): an Ollama tag is *uncurated*, so the
# reasoner logs an expected `reasoner.model.uncurated` warning. `ollama` is a
# named endpoint with its own URL, dialect and cold-start timeout, so no
# DIALECT is needed:
#
#   export OPENRAL_REASONER_MODEL=qwen3:8b
#   export OPENRAL_REASONER_ENDPOINT=ollama
#
# Flags:
#   --no-pull           Skip the `ollama pull` step (binary install only).
#   --model <tag>       Pull a different model tag (default: qwen3:8b).

# Linux install pins. The installer script is fetched from the release tag (not
# the moving ollama.com/install.sh), checked against OLLAMA_INSTALL_SH_SHA256,
# and told to install exactly OLLAMA_VERSION. Bump both together: download
# https://raw.githubusercontent.com/ollama/ollama/v<ver>/scripts/install.sh,
# review it, and pin its `sha256sum`. Empty = refuse to install (fail closed).
OLLAMA_VERSION="${OLLAMA_VERSION:-0.32.14}"
OLLAMA_INSTALL_SH_SHA256="${OLLAMA_INSTALL_SH_SHA256:-}"  # TODO(maintainer): pin for v0.32.14

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

Afterwards, export these to wire the reasoner at the local endpoint
(model-first contract, ADR-0088):
  export OPENRAL_REASONER_MODEL=qwen3:8b
  export OPENRAL_REASONER_ENDPOINT=ollama
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
      if [[ ! "${OLLAMA_INSTALL_SH_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "refusing to run an unpinned installer: set OLLAMA_INSTALL_SH_SHA256 to the" >&2
        echo "reviewed sha256 of ollama v${OLLAMA_VERSION}'s scripts/install.sh, or install" >&2
        echo "ollama by hand (https://ollama.com/download) and re-run." >&2
        exit 1
      fi
      installer="$(mktemp)"
      trap 'rm -f "${installer}"' EXIT
      echo "==> installing ollama v${OLLAMA_VERSION} via its release-tag installer (sha256-pinned)"
      curl --proto '=https' --tlsv1.2 -fsSL -o "${installer}" \
        "https://raw.githubusercontent.com/ollama/ollama/v${OLLAMA_VERSION}/scripts/install.sh"
      echo "${OLLAMA_INSTALL_SH_SHA256}  ${installer}" | sha256sum -c -
      OLLAMA_VERSION="${OLLAMA_VERSION}" sh "${installer}"
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
  # A private mktemp file, not a fixed /tmp name another local user could pre-create.
  OLLAMA_LOG="$(mktemp -t ollama-serve.XXXXXX.log)"
  nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  echo "    log: ${OLLAMA_LOG}"
  # Wait up to ~5 s for the listener to come up before giving up.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    if probe_port localhost 11434; then break; fi
  done
  if ! probe_port localhost 11434; then
    echo "ollama serve failed to bind on localhost:11434 — see ${OLLAMA_LOG}" >&2
    exit 1
  fi
fi

# 3. Pull the baseline model (idempotent — ollama re-uses cached layers).
if [[ "${DO_PULL}" -eq 1 ]]; then
  echo "==> pulling ${MODEL} (will reuse layers if already cached)"
  ollama pull "${MODEL}"
fi

cat <<EOF

==> Local reasoner baseline ready. Export these to wire it up
    (model-first contract, ADR-0088 — an Ollama tag is uncurated, so the
    reasoner logs an expected reasoner.model.uncurated warning; the named
    'ollama' endpoint supplies the URL and dialect):

    export OPENRAL_REASONER_MODEL=${MODEL}
    export OPENRAL_REASONER_ENDPOINT=ollama

Then 'openral doctor' reports "Reasoner LLM" as warn (uncurated model —
expected) and "Reasoner endpoint" as ok.
EOF
