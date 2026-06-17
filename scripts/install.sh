#!/usr/bin/env bash
#
# OpenRAL — Tier-0 curl-bash installer (ADR-0021).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OpenRAL/openral/master/scripts/install.sh | bash
#
# What this does (Tier-0 only — no sudo, no apt, no GPU, no ROS 2):
#   1. Detect OS / arch / shell.
#   2. Install `uv` if missing (the astral.sh installer; user-local).
#   3. Install CPython 3.12 via `uv python install 3.12`  (user-local).
#   4. `uv tool install --python 3.12 openral-cli`        — drops `openral`
#                                                            into ~/.local/bin/.
#   5. Print PATH instructions if ~/.local/bin is not on $PATH.
#   6. Print the "next steps" menu for the heavier opt-in groups
#      (`openral install sim|libero|metaworld|ros|…`).
#
# What this DOES NOT do (by design — see ADR-0021 / CLAUDE.md §4):
#   - Install ROS 2 (sudo + apt).            Use: openral install ros
#   - Install MuJoCo / LIBERO / RoboCasa.    Use: openral install sim|libero|robocasa
#   - Install NVIDIA drivers / CUDA.         Out of scope.
#   - Run as root.                            Refused.
#
# Environment knobs:
#   OPENRAL_INSTALL_SOURCE   pypi (default) | git+https://github.com/OpenRAL/openral
#   OPENRAL_INSTALL_VERSION  PyPI version specifier (default: empty → latest).
#   OPENRAL_INSTALL_INDEX    extra package index, appended as uv
#                            `--extra-index-url` (e.g.
#                            https://test.pypi.org/simple/ to install a
#                            TestPyPI build). `openral-*` resolve from here;
#                            third-party deps still come from real PyPI.
#   OPENRAL_INSTALL_DEBUG    1 → set -x.

set -euo pipefail

if [[ "${OPENRAL_INSTALL_DEBUG:-}" == "1" ]]; then
    set -x
fi

# ── Helpers ────────────────────────────────────────────────────────────────────

_color() {
    # $1=ansi $2=text   (no-op when stdout is not a TTY)
    if [[ -t 1 ]]; then printf '\033[%sm%s\033[0m\n' "$1" "$2"; else printf '%s\n' "$2"; fi
}
info()  { _color "0;36" "==> $*"; }
warn()  { _color "0;33" "warn: $*" >&2; }
fatal() { _color "0;31" "error: $*" >&2; exit 1; }

# ── Pre-flight ─────────────────────────────────────────────────────────────────

if [[ "$(id -u)" == "0" ]]; then
    fatal "do not run this installer as root — it installs to your user-local ~/.local/."
fi

OS="$(uname -s)"
case "${OS}" in
    Linux)  PLATFORM="linux"  ;;
    Darwin) PLATFORM="darwin" ;;
    *)      fatal "unsupported OS: ${OS} (supported: Linux, Darwin)" ;;
esac
info "detected platform: ${PLATFORM} $(uname -m)"

# ── 1. Install uv if missing ───────────────────────────────────────────────────

if ! command -v uv >/dev/null 2>&1; then
    info "uv not found — installing from https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer drops the binary into one of these per its own logic.
    for candidate in "${HOME}/.local/bin" "${HOME}/.cargo/bin"; do
        if [[ -x "${candidate}/uv" ]]; then
            export PATH="${candidate}:${PATH}"
            break
        fi
    done
    command -v uv >/dev/null 2>&1 || fatal "uv installation finished but \`uv\` is still not on \$PATH."
fi
info "uv: $(command -v uv) ($(uv --version))"

# ── 2. Ensure CPython 3.12 is available (uv-managed, no apt needed) ────────────

info "installing CPython 3.12 (uv-managed; pyproject.toml pins >=3.12,<3.13)"
uv python install 3.12

# ── 3. uv tool install openral-cli ─────────────────────────────────────────────

OPENRAL_INSTALL_SOURCE="${OPENRAL_INSTALL_SOURCE:-pypi}"
OPENRAL_INSTALL_VERSION="${OPENRAL_INSTALL_VERSION:-}"

case "${OPENRAL_INSTALL_SOURCE}" in
    pypi)
        if [[ -n "${OPENRAL_INSTALL_VERSION}" ]]; then
            spec="openral-cli==${OPENRAL_INSTALL_VERSION}"
        else
            spec="openral-cli"
        fi
        ;;
    git+*)
        # e.g. git+https://github.com/OpenRAL/openral
        spec="openral-cli @ ${OPENRAL_INSTALL_SOURCE#git+}#subdirectory=python/cli"
        ;;
    *)
        fatal "unknown OPENRAL_INSTALL_SOURCE: ${OPENRAL_INSTALL_SOURCE} (expected: pypi | git+https://…)"
        ;;
esac

# Optional extra index (e.g. TestPyPI). uv's default first-index strategy
# resolves `openral-*` from this index while third-party deps still come from
# real PyPI, since the openral names are absent from PyPI until the namespace
# lands. Empty-array expansion is written `${arr[@]+"${arr[@]}"}` so it is
# safe under `set -u` on bash 3.2 (macOS).
extra_index_args=()
if [[ -n "${OPENRAL_INSTALL_INDEX:-}" ]]; then
    info "extra index: ${OPENRAL_INSTALL_INDEX}"
    extra_index_args=(--extra-index-url "${OPENRAL_INSTALL_INDEX}")
fi

info "installing ${spec} (uv tool install)"
# --force re-installs even when the tool is already present; matches the
# `curl … | bash` muscle memory of "running it again gives me the latest".
uv tool install --force --python 3.12 \
    ${extra_index_args[@]+"${extra_index_args[@]}"} "${spec}"

# ── 4. PATH guidance ───────────────────────────────────────────────────────────

# `uv tool install` prints its own PATH instructions; we re-emit a compact
# version here so the user sees them even when uv decided everything was fine.
TOOL_BIN="$(uv tool dir --bin 2>/dev/null || echo "${HOME}/.local/bin")"
case ":${PATH}:" in
    *":${TOOL_BIN}:"*) ;;
    *)
        warn "${TOOL_BIN} is not on your \$PATH. Add this line to ~/.bashrc / ~/.zshrc:"
        printf '  export PATH="%s:$PATH"\n' "${TOOL_BIN}"
        ;;
esac

# ── 5. Verify ──────────────────────────────────────────────────────────────────

if command -v openral >/dev/null 2>&1; then
    info "openral installed: $(command -v openral)"
else
    warn "openral is installed but not yet on \$PATH (see PATH guidance above)."
    warn "Run \`openral doctor\` after restarting your shell."
fi

# ── 6. Next steps ──────────────────────────────────────────────────────────────

cat <<'EOF'

==> Tier-0 install complete.

Quick start:
  openral doctor                  # diagnose the host (Python, OS, GPU, USB)
  openral --help                  # list every subcommand

Opt-in extras (installed into the same managed venv):
  openral install sim             # gym-aloha + gym-pusht + MuJoCo (CPU sim)
  openral install libero          # LIBERO task suite (mutually exclusive with robocasa)
  openral install metaworld       # MetaWorld MT50 (Sawyer)
  openral install maniskill3      # SAPIEN GPU physics
  openral install simpler-env     # real-to-sim correlator
  openral install robocasa        # RoboCasa kitchens (excludes libero — ADR-0011)
  openral install rldx            # RLDX-1 sidecar client
  openral install list            # show every known group

System bootstrap (needs sudo + clones the openral repo):
  openral install ros             # ROS 2 + libusb + udev (apt)

Notes:
  - Bare `openral` (no args) drops into the interactive REPL; pass a
    subcommand (e.g. `openral doctor`) for one-shot mode in scripts / CI.
  - Re-run this installer at any time to upgrade to the latest openral-cli.

EOF
