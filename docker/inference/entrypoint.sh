#!/usr/bin/env bash
# OpenRAL deploy-image entrypoint — probes for ROS 2 and the local
# colcon `install/` overlay, sources whichever it finds, then exec's
# the user command.
#
# Probe-style (rather than hardcoded `source /opt/ros/jazzy/setup.bash`)
# so the same script survives future Dockerfiles that ship a different
# ROS distro (Humble on L4T 22.04, none on a minimal variant, etc.) —
# none of those exist today, but adding the dispatch up-front beats
# re-implementing the lookup in every per-arch image. ADR-0010
# amendment "Single-Dockerfile consolidation + CUDA-13/DeepStream-9
# alignment".
#
# The local `/workspace/install/setup.bash` overlay is the colcon
# install tree baked into the image by the builder stage — it carries
# openral_msgs (IDL), every openral_* lifecycle node, and the C++
# safety_kernel binary. Sourcing it AFTER the system ROS distro keeps
# the ament search path correct (system first, overlay second).
set -eu
for setup in /opt/ros/*/setup.bash; do
    if [ -f "$setup" ]; then
        # `setup.bash` is not strict-bash-safe; relax -u inside the source.
        set +u
        # shellcheck disable=SC1090
        source "$setup"
        set -u
        break
    fi
done
if [ -f /workspace/install/setup.bash ]; then
    set +u
    # shellcheck disable=SC1091
    source /workspace/install/setup.bash
    set -u
fi
exec "$@"
