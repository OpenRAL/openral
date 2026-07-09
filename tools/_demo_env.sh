# Sourceable env for the sim+dashboard demo recordings (see
# docs/superpowers/specs/2026-06-12-sim-dashboard-demo-recordings-design.md).
# Reproduces the hard-won robocasa deploy-sim recipe env. Source, don't exec.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Strip miniforge/conda from PATH (else python3.13 libpython contaminates the
#    colcon overlay / ROS node spawn — recipe step 2).
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i 'miniforge\|conda\|mamba' | paste -sd: -)"
export PATH

# 2. ROS base + this repo's colcon overlay.
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"

# 3. venv site-packages on PYTHONPATH so spawned ROS nodes import torch /
#    transformers / robosuite (recipe step 4).
VENV_SP="$REPO_ROOT/.venv/lib/python3.12/site-packages"
export PYTHONPATH="$VENV_SP${PYTHONPATH:+:$PYTHONPATH}"

# 4. Shared graph domain so the dispatch process discovers /openral/execute_rskill.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# 5. robocasa "install on first use" must not block on an interactive prompt.
export OPENRAL_AUTO_INSTALL_DEPS=1

# 6. 8 GB-GPU headroom: expandable segments reduces the first-forward spike OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 7. GUI on the user's X display for the viewer + dashboard.
export DISPLAY="${DISPLAY:-:1}"

export OPENRAL="$REPO_ROOT/.venv/bin/openral"
