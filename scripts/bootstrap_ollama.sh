#!/usr/bin/env bash
set -euo pipefail

# Idempotent Ollama bootstrap for the OpenRAL local reasoner baseline.
#
# Installs the `ollama` binary (Linux: upstream's install.sh at a pinned
# release tag, run only after its sha256 matches the pin below; macOS:
# `brew install`), starts the daemon if it isn't already running, and
# pulls the default model: qwen3.5:4b, the curated `qwen3.5-4b` reasoner
# (vision + tools, ~3.4 GB; `--model qwen3.5:9b` for the curated 9B), so one
# variable selects it:
#
#   export OPENRAL_REASONER_MODEL=qwen3.5-4b
#
# Any other tag is *uncurated* (ADR-0088): the reasoner logs an expected
# `reasoner.model.uncurated` warning, and the named `ollama` endpoint supplies
# the URL, dialect and cold-start timeout:
#
#   export OPENRAL_REASONER_MODEL=<tag>
#   export OPENRAL_REASONER_ENDPOINT=ollama
#
# The reasoner's prompt is 9-26K tokens; Ollama's default window on a GPU
# under 24 GiB is 4096, and it truncates silently (the client refuses that).
# A daemon this script starts, or the systemd unit its install registers, gets
# OLLAMA_CONTEXT_LENGTH=32768; an Ollama already installed keeps its own setting
# (`sudo systemctl edit ollama`: `Environment=OLLAMA_CONTEXT_LENGTH=32768`).
#
# Flags:
#   --no-pull           Skip the `ollama pull` step (binary install only).
#   --model <tag>       Pull a different model tag (default: qwen3.5:4b).

# Linux install pins. The installer script is fetched from the release tag (not
# the moving ollama.com/install.sh), checked against OLLAMA_INSTALL_SH_SHA256,
# and told to install exactly OLLAMA_VERSION. Bump both together: download
# https://raw.githubusercontent.com/ollama/ollama/v<ver>/scripts/install.sh,
# review it, and pin its `sha256sum`. Empty or a mismatch = refuse (fail closed).
OLLAMA_VERSION="${OLLAMA_VERSION:-0.32.14}"
# sha256 of v0.32.14's scripts/install.sh, computed from the tag on 2026-11-11.
OLLAMA_INSTALL_SH_SHA256="${OLLAMA_INSTALL_SH_SHA256:-25f64b810b947145095956533e1bdf56eacea2673c55a7e586be4515fc882c9f}"

MODEL="qwen3.5:4b"
DO_PULL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pull) DO_PULL=0; shift ;;
    --model)   MODEL="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bootstrap_ollama.sh [--no-pull] [--model <tag>]

Installs Ollama (if missing), starts the daemon with a 32K window, and pulls
the curated local reasoner model (qwen3.5:4b by default).

  --no-pull         Skip the `ollama pull` step (binary install only).
  --model <tag>     Pull a different model tag (default: qwen3.5:4b).

Afterwards, select it (model-first contract, ADR-0088):
  export OPENRAL_REASONER_MODEL=qwen3.5-4b
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
      # The installer registers and starts a systemd `ollama` unit on Ollama's
      # 4096-token default; the unit this script installed gets the 32K window.
      if command -v systemctl >/dev/null 2>&1 && systemctl cat ollama.service >/dev/null 2>&1; then
        echo "==> setting OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32768} on the ollama systemd unit"
        sudo mkdir -p /etc/systemd/system/ollama.service.d
        printf '[Service]\nEnvironment=OLLAMA_CONTEXT_LENGTH=%s\n' "${OLLAMA_CONTEXT_LENGTH:-32768}" |
          sudo tee /etc/systemd/system/ollama.service.d/openral-context.conf >/dev/null
        sudo systemctl daemon-reload
        sudo systemctl restart ollama
        OLLAMA_UNIT_SET=1
      fi
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

if [[ "${OLLAMA_UNIT_SET:-0}" -eq 1 ]]; then
  # The unit was just restarted: wait for it, never race it with a second serve.
  for _ in $(seq 60); do probe_port localhost 11434 && break; sleep 0.5; done
  if ! probe_port localhost 11434; then
    echo "the ollama systemd unit did not start listening on localhost:11434 within 30 s;" >&2
    echo "see: systemctl status ollama; journalctl -u ollama" >&2
    exit 1
  fi
fi

if [[ "${OLLAMA_UNIT_SET:-0}" -eq 1 ]] && probe_port localhost 11434; then
  echo "==> ollama systemd unit serving on localhost:11434 with the 32K window"
elif probe_port localhost 11434; then
  echo "==> ollama daemon already serving on localhost:11434"
  # Not started here (e.g. the systemd unit Ollama's installer registers), so it
  # runs Ollama's 4096-token default unless its owner raised it.
  echo "WARNING: this daemon was not started by this script, so it keeps its own context" >&2
  echo "         window (Ollama's default is 4096 tokens; the reasoner's prompt is 9-26K and" >&2
  echo "         the client refuses a truncated prompt). For the systemd unit:" >&2
  echo "           sudo systemctl edit ollama   # Environment=OLLAMA_CONTEXT_LENGTH=32768" >&2
  echo "           sudo systemctl restart ollama" >&2
else
  echo "==> starting ollama serve in the background"
  # A private mktemp file, not a fixed /tmp name another local user could pre-create.
  OLLAMA_LOG="$(mktemp -t ollama-serve.XXXXXX.log)"
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}" nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
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

if [[ "${MODEL}" == "qwen3.5:4b" || "${MODEL}" == "qwen3.5:9b" ]]; then
  cat <<EOF

==> Local reasoner ready. Select the curated model (ADR-0088):

    export OPENRAL_REASONER_MODEL=${MODEL/:/-}
EOF
else
  cat <<EOF

==> Local reasoner ready. ${MODEL} is uncurated, so the reasoner logs an
    expected reasoner.model.uncurated warning; the named 'ollama' endpoint
    supplies the URL and dialect:

    export OPENRAL_REASONER_MODEL=${MODEL}
    export OPENRAL_REASONER_ENDPOINT=ollama
EOF
fi
