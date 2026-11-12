#!/usr/bin/env bash
set -euo pipefail

# Idempotent Ollama bootstrap for the OpenRAL local reasoner baseline.
#
# Installs the `ollama` binary (Linux: the pinned release's tarballs, each
# extracted only after its sha256 matches the pin below; macOS: `brew install`),
# starts the daemon if it isn't already running, and pulls the default model:
# the curated Ollama reasoner with the smallest GPU floor in
# openral_core.REASONER_MODELS (qwen3.5:4b today), so one variable selects it:
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
# A daemon this script starts, or the systemd unit it installs, gets
# OLLAMA_CONTEXT_LENGTH=32768; an Ollama already installed keeps its own setting
# (`sudo systemctl edit ollama`: `Environment=OLLAMA_CONTEXT_LENGTH=32768`).
#
# Flags:
#   --no-pull           Skip the `ollama pull` step (binary install only).
#   --model <tag>       Pull a different model tag (default: the 4B above).
#
# Environment:
#   OLLAMA_PREFIX       Install prefix (default /usr/local, which also registers
#                       the systemd unit; any other prefix installs the files only).
#   OLLAMA_HOST         host:port the daemon serves on (default 127.0.0.1:11434).
#   OPENRAL_PYTHON      A Python that imports openral_core (default: the repo's
#                       .venv, else python3).

# Linux install pins: the exact release artifacts, by sha256 (the digests GitHub
# lists for the v0.32.14 release assets, re-checked by download on 2026-11-12).
# Each tarball is downloaded from the GitHub release and verified, and nothing is
# extracted until every one matches; an unpinned artifact or a mismatch refuses
# (fail closed). To bump, change OLLAMA_VERSION and every pin together, from
#   gh release view v<ver> -R ollama/ollama --json assets
# and re-measure the curated Ollama models on it (packages/openral_reasoner_ros/README.md).
OLLAMA_VERSION="0.32.14"
ollama_sha256() {
  case "$1" in
    ollama-linux-amd64) echo c620917a71e146ab3a7f893084f066069c4c65d144ef8379a91c3cbe8b27de8f ;;
    ollama-linux-arm64) echo 7802b739fbdc74df556600f1619f86457b69dce913301cf2d91f7f9d7f7a41b8 ;;
    ollama-linux-arm64-jetpack5) echo 6d1b91a3d049d201b2dd8f3b8ee553bee64114d6feb5d7dc131f206bbef7d61b ;;
    ollama-linux-arm64-jetpack6) echo 5ef2948dce02c7edbfca83de5967e064ddeade86dce0afcbd9040284eba23f07 ;;
    *) echo "" ;;
  esac
}
OLLAMA_PREFIX="${OLLAMA_PREFIX:-/usr/local}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST#http://}"
export OLLAMA_HOST # `ollama serve` and `ollama pull` read it
OLLAMA_PORT="${OLLAMA_HOST##*:}"
OLLAMA_ADDR="${OLLAMA_HOST%:*}"

# The curated Ollama models, read from the registry (the single source of truth):
# one "<ollama tag> <reasoner model id>" line each, smallest GPU floor first.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${OPENRAL_PYTHON:-}" ]]; then
  OPENRAL_PYTHON=python3
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then OPENRAL_PYTHON="${REPO_ROOT}/.venv/bin/python"; fi
fi
CURATED="$("${OPENRAL_PYTHON}" - <<'PY' || true
from openral_core import OLLAMA_BASE_URL, REASONER_MODELS

ollama = [m for m in REASONER_MODELS.values() if m.default_endpoint == OLLAMA_BASE_URL]
for m in sorted(ollama, key=lambda m: m.min_gpu_vram_gb or 0.0):
    print(m.served_model_id, m.id)
PY
)"
if [[ -z "${CURATED}" ]]; then
  echo "could not read the curated Ollama models from openral_core.REASONER_MODELS with" >&2
  echo "${OPENRAL_PYTHON}: run 'just sync' (or set OPENRAL_PYTHON) and re-run." >&2
  exit 1
fi
# The reasoner model id curated for an Ollama tag, or "" when it is uncurated.
curated_id() { awk -v tag="$1" '$1 == tag { print $2 }' <<<"${CURATED}"; }

MODEL="$(head -n1 <<<"${CURATED}" | cut -d' ' -f1)"
DO_PULL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pull) DO_PULL=0; shift ;;
    --model)   MODEL="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: bootstrap_ollama.sh [--no-pull] [--model <tag>]

Installs Ollama v${OLLAMA_VERSION} (if missing), starts the daemon with a 32K
window, and pulls a curated local reasoner model (${MODEL} by default).

  --no-pull         Skip the 'ollama pull' step (binary install only).
  --model <tag>     Pull a different model tag (default: ${MODEL}).

Curated Ollama models (openral_core.REASONER_MODELS, <tag> <model id>):
$(sed 's/^/  /' <<<"${CURATED}")

Afterwards, select it (model-first contract, ADR-0088):
  export OPENRAL_REASONER_MODEL=$(curated_id "${MODEL}")
  export OPENRAL_REASONER_ENDPOINT=ollama   # if another endpoint is exported
EOF
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

OS="$(uname -s)"

# Linux: download every release tarball this host needs, verify each against its
# pin, and extract them into OLLAMA_PREFIX only once all of them match. Replaces
# upstream's install.sh, which streams the tarball from ollama.com into tar
# unverified. Not done here (do it by hand if needed): GPU driver installs, and the
# ROCm runtime for an AMD GPU.
install_ollama_linux() {
  local arch sudo="" dl n pin
  case "$(uname -m)" in
    x86_64) arch=amd64 ;;
    aarch64 | arm64) arch=arm64 ;;
    *) echo "unsupported CPU $(uname -m): install ollama by hand (https://ollama.com/download)" >&2; exit 1 ;;
  esac
  local names=("ollama-linux-${arch}")
  if [[ -f /etc/nv_tegra_release ]]; then
    if grep -q R36 /etc/nv_tegra_release; then
      names+=("ollama-linux-${arch}-jetpack6")
    elif grep -q R35 /etc/nv_tegra_release; then
      names+=("ollama-linux-${arch}-jetpack5")
    else
      echo "WARNING: unsupported JetPack release; ollama may not use the GPU" >&2
    fi
  fi
  if [[ -e /dev/kfd ]]; then
    echo "WARNING: an AMD GPU is present; this script does not install ollama's ROCm runtime" >&2
  fi
  command -v zstd >/dev/null 2>&1 || { echo "zstd is required: sudo apt-get install zstd" >&2; exit 1; }

  dl="$(mktemp -d)"
  # Expanded now: `dl` is local, gone by the time the EXIT trap runs.
  # shellcheck disable=SC2064
  trap "rm -rf '${dl}'" EXIT
  for n in "${names[@]}"; do
    pin="$(ollama_sha256 "${n}")"
    if [[ ! "${pin}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "refusing to install ${n} v${OLLAMA_VERSION}: no sha256 pinned for it" >&2
      exit 1
    fi
    echo "==> downloading ${n} v${OLLAMA_VERSION}"
    curl --proto '=https' --tlsv1.2 -fL --progress-bar -o "${dl}/${n}.tar.zst" \
      "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/${n}.tar.zst"
    if ! echo "${pin}  ${dl}/${n}.tar.zst" | sha256sum -c --quiet -; then
      echo "refusing to install ${n} v${OLLAMA_VERSION}: its sha256 does not match the pin" >&2
      exit 1
    fi
  done

  mkdir -p "${OLLAMA_PREFIX}" 2>/dev/null || true
  if [[ ! -w "${OLLAMA_PREFIX}" ]]; then sudo=sudo; fi
  echo "==> installing ollama v${OLLAMA_VERSION} into ${OLLAMA_PREFIX} (sha256-verified)"
  ${sudo} rm -rf "${OLLAMA_PREFIX}/lib/ollama"
  ${sudo} install -d -m755 "${OLLAMA_PREFIX}/bin" "${OLLAMA_PREFIX}/lib/ollama"
  for n in "${names[@]}"; do
    zstd -dc "${dl}/${n}.tar.zst" | ${sudo} tar -xf - -C "${OLLAMA_PREFIX}"
  done
  OLLAMA_BIN="${OLLAMA_PREFIX}/bin/ollama"

  # The systemd unit (as upstream's installer writes it), with the 32K window.
  if [[ "${OLLAMA_PREFIX}" == /usr/local ]] && command -v systemctl >/dev/null 2>&1 \
    && [[ "$(systemctl is-system-running 2>/dev/null || true)" =~ ^(running|degraded)$ ]]; then
    echo "==> registering the ollama systemd unit with OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32768}"
    id ollama >/dev/null 2>&1 || sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
    for g in render video; do
      if getent group "${g}" >/dev/null 2>&1; then sudo usermod -a -G "${g}" ollama; fi
    done
    sudo usermod -a -G ollama "$(whoami)"
    sudo tee /etc/systemd/system/ollama.service >/dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=${OLLAMA_BIN} serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=${PATH}"
Environment="OLLAMA_HOST=${OLLAMA_HOST}"
Environment="OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32768}"

[Install]
WantedBy=default.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    sudo systemctl restart ollama
    OLLAMA_UNIT_SET=1
  fi
}

# 1. Install the ollama binary if missing.
if OLLAMA_BIN="$(command -v ollama)"; then
  echo "==> ollama already installed: ${OLLAMA_BIN}"
elif [[ -x "${OLLAMA_PREFIX}/bin/ollama" ]]; then
  OLLAMA_BIN="${OLLAMA_PREFIX}/bin/ollama"
  echo "==> ollama already installed: ${OLLAMA_BIN}"
else
  case "${OS}" in
    Linux) install_ollama_linux ;;
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        echo "==> installing ollama via brew"
        brew install ollama
        OLLAMA_BIN="$(command -v ollama)"
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
#    (the unit above), on macOS it's started by the menu-bar app. We
#    only background a `serve` if no listener answers on OLLAMA_HOST.
probe_port() {
  python3 - "${OLLAMA_ADDR}" "${OLLAMA_PORT}" <<'PY'
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
  for _ in $(seq 60); do probe_port && break; sleep 0.5; done
  if ! probe_port; then
    echo "the ollama systemd unit did not start listening on ${OLLAMA_HOST} within 30 s;" >&2
    echo "see: systemctl status ollama; journalctl -u ollama" >&2
    exit 1
  fi
  echo "==> ollama systemd unit serving on ${OLLAMA_HOST} with OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32768}"
elif probe_port; then
  echo "==> ollama daemon already serving on ${OLLAMA_HOST}"
  # Not started here (e.g. a systemd unit an earlier install registered), so it
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
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}" nohup "${OLLAMA_BIN}" serve >"${OLLAMA_LOG}" 2>&1 &
  echo "    log: ${OLLAMA_LOG}"
  # Wait up to ~5 s for the listener to come up before giving up.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    if probe_port; then break; fi
  done
  if ! probe_port; then
    echo "ollama serve failed to bind on ${OLLAMA_HOST} — see ${OLLAMA_LOG}" >&2
    exit 1
  fi
fi

# 3. Pull the baseline model (idempotent — ollama re-uses cached layers).
if [[ "${DO_PULL}" -eq 1 ]]; then
  echo "==> pulling ${MODEL} (will reuse layers if already cached)"
  "${OLLAMA_BIN}" pull "${MODEL}"
fi

MODEL_ID="$(curated_id "${MODEL}")"
if [[ -n "${MODEL_ID}" ]]; then
  cat <<EOF

==> Local reasoner ready. Select the curated model (ADR-0088):

    export OPENRAL_REASONER_MODEL=${MODEL_ID}
    export OPENRAL_REASONER_ENDPOINT=ollama   # overrides an endpoint set for another model
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
