# Default task
default:
    @just --list

# Re-running after a `git pull` is the supported re-sync flow: uv sync and
# colcon are incremental, so the 2nd run finishes in seconds. When the REPL
# exits the subshell exits and the original shell is untouched.
# Fresh-clone bring-up: bootstrap (skipped if done) → uv sync → ros2-build → openral REPL.
quickstart:
    @# Cheap bootstrap probe: skip "just bootstrap" if the venv exists AND
    @# a ROS 2 distro is installed under /opt/ros/. Either signal missing
    @# → run the full bootstrap (sudo + apt + ROS 2 keyring on Linux).
    @if [ -x .venv/bin/python ] && ls /opt/ros/*/setup.bash >/dev/null 2>&1; then \
        echo '==> quickstart: .venv + /opt/ros detected, skipping just bootstrap'; \
    else \
        echo '==> quickstart: running just bootstrap (fresh clone or no ROS 2)'; \
        just bootstrap; \
    fi
    @echo '==> quickstart: just sync --all-packages (incremental, with hf-libero repair)'
    @# Route through `just sync` (not bare `uv sync`) so the hf-libero
    @# distutils-uninstall trap is repaired before+after. Without this,
    @# a venv that previously saw `uv sync --group libero` (or `--group
    @# robocasa`, etc.) trips on `Unable to uninstall hf-libero==0.1.3`
    @# and bails out installing nothing — leaving the workspace without
    @# any `openral-*` member and the REPL launch with no `openral_core`.
    just sync --all-packages
    @echo '==> quickstart: just ros2-build (incremental colcon rebuild)'
    just ros2-build
    @echo '==> quickstart: installing openral CLI to ~/.local/bin'
    just install-cli
    @echo '==> quickstart: sourcing install/setup.bash and launching openral REPL'
    @echo '    (exit the REPL with Ctrl-D or "exit" to return to your shell)'
    @# Clean stale Fast-DDS SHM lockfiles before launching the REPL.
    @# Jazzy defaults to Fast-DDS, whose /dev/shm/fastrtps_* port-lock
    @# files persist after unclean exits and cause
    @# "[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7000"
    @# on the next launch. ``openral deploy sim`` runs the same cleanup
    @# internally, but doing it here too means ad-hoc ``ros2 topic
    @# list`` in the REPL works on a fresh-clone host.
    @find /dev/shm -maxdepth 1 -name 'fastrtps_*' -user "$(whoami)" -delete 2>/dev/null || true
    @exec bash -c 'source install/setup.bash && exec .venv/bin/openral'

# One-shot bootstrap for a fresh dev machine
bootstrap:
    @case "$(uname)" in \
        Linux)  ./scripts/bootstrap_ubuntu.sh ;; \
        Darwin) ./scripts/bootstrap_macos.sh ;; \
        *) echo "Unsupported OS: $(uname)" >&2; exit 1 ;; \
    esac
    uv sync --all-packages
    @just install-hooks
    @echo ""
    @echo "==> uv environment ready at .venv/"
    @echo "==> Run tools via 'uv run <cmd>' (e.g. 'uv run openral doctor', 'just test'),"
    @echo "    or activate the venv directly with: source .venv/bin/activate"
    @ls -1d /opt/ros/*/setup.bash 2>/dev/null | head -1 | awk '{ if ($0) print "==> For ROS 2 commands also: source " $0 }'

# Point git at the in-repo hooks (.githooks/) and build the pre-commit
# environments. Installs BOTH: DCO auto-sign-off (prepare-commit-msg) and the
# pre-commit framework — .githooks/pre-commit + .githooks/commit-msg delegate to
# `pre-commit run` for ruff, ruff-format, mypy, codespell, clang-format, and the
# conventional-commit check. We can't use `pre-commit install` because it
# refuses to run with core.hooksPath set, so the wrappers are committed and this
# recipe just pre-builds their environments. Idempotent; run by `just bootstrap`.
install-hooks:
    @git config core.hooksPath .githooks
    @uv run pre-commit install-hooks
    @echo "==> git hooks installed (.githooks): DCO auto-sign-off + pre-commit (ruff, ruff-format, mypy, codespell, conventional-commit)"

# Write ~/.local/bin/openral so users can run `openral` (or `openral <cmd>`)
# from any terminal without `just`. Also patches ~/.bashrc / ~/.zshrc to add
# ~/.local/bin to $PATH when it isn't already there. Idempotent — re-run
# after moving the repo or cloning a fresh copy.
install-cli:
    @python3 scripts/install_cli.py

# Install Ollama + pull the local reasoner baseline model (qwen3:8b).
# Opt-in: deliberately not part of `just bootstrap` because the model
# pull is several GB (CLAUDE.md §1.4 — no hidden network I/O).
bootstrap-ollama *args:
    ./scripts/bootstrap_ollama.sh {{args}}

# Canonical `uv sync` wrapper that also repairs the malformed
# `hf-libero==0.1.3` install. Use this instead of calling `uv sync`
# directly when switching dependency groups (libero ⊥ robocasa, etc.)
# or anywhere `uv sync` would otherwise fail with::
#
#   error: Unable to uninstall `hf-libero==0.1.3`.
#   distutils-installed distributions do not include the metadata
#   required to uninstall safely.
#
# The hf-libero sdist installs a spurious top-level `.egg-info` FILE
# alongside the proper `.dist-info/` directory; uv then sees both the
# modern + legacy markers and falls back to the distutils uninstall
# path, which fails. `scripts/repair_hf_libero_install.py` removes the
# bogus file + the matching RECORD line so the next `uv sync` can
# uninstall hf-libero cleanly. Idempotent — no-op when hf-libero
# isn't installed.
#
# Pass any flag you'd normally pass to `uv sync` (e.g.
# `--group libero`, `--reinstall-package foo`). ``--all-packages`` is
# forced in automatically (see below) — you no longer pass it yourself.
#
# Examples:
#   just sync                       # whole workspace, no optional groups
#   just sync --group libero        # + libero extras (still all-packages)
#   just sync --group robocasa      # + robocasa extras (still all-packages)
#   CC=/usr/bin/gcc just sync --group libero
#
# ``--all-packages`` is forced unless the caller scoped the sync to a
# single member (``--package`` / ``-p``). WITHOUT it, ``uv sync --group
# <X>`` syncs only the workspace ROOT and UNINSTALLS every editable
# ``openral-*`` member — leaving the REPL / ``openral deploy sim`` with no
# ``openral_core`` (a `ModuleNotFoundError` that looks unrelated to the
# group switch that caused it). The recipe used to pass ``{{args}}``
# verbatim, so ``just sync --group robocasa`` was a silent venv-wrecker;
# forcing ``--all-packages`` makes every group switch safe by default.
#
# Two repair calls (before + after) are both necessary:
#   * BEFORE — if a previous sync left hf-libero in the broken state,
#     this one would refuse to uninstall it. Repair first so the
#     ensuing `uv sync` can swap groups cleanly.
#   * AFTER — if this sync just installed hf-libero (e.g. --group
#     libero), the broken state is recreated; pre-emptively repair so
#     the NEXT `just sync --group <other>` doesn't trip on it.
sync *args:
    @command -v python3 >/dev/null && python3 scripts/repair_hf_libero_install.py || true
    @# Force --all-packages unless the caller already set it or scoped to a
    @# single member (--package/-p), so a bare `just sync --group <X>` can
    @# never silently uninstall the editable `openral-*` workspace members.
    @case " {{args}} " in \
        *" --all-packages "*|*" --package "*|*" -p "*) uv sync {{args}} ;; \
        *) uv sync --all-packages {{args}} ;; \
    esac
    @uv run --no-sync python scripts/repair_hf_libero_install.py

# Lint everything
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy --strict -p openral_core -p openral_cli -p openral_sim -p openral_observability -p openral_runner -p openral_reasoner -p openral_wam
    uv run mypy --strict tools/

# Format
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Build the consolidated x86 deploy image — Ubuntu 24.04 + Py 3.12 +
# CUDA 13 + ROS 2 Jazzy + GStreamer (ADR-0010 amendment "Single-Dockerfile
# consolidation + CUDA-13/DeepStream-9 alignment"). This is the ONLY
# supported deployment surface today; the per-flavour Dockerfiles
# `.x86-ros`, `.x86-deepstream`, `.l4t` of PR #93 have been folded into
# this one or dropped. ENTRYPOINT is `/entrypoint.sh openral deploy`.
docker-build-x86:
    docker buildx build -f docker/inference/Dockerfile.x86 \
        -t openral:x86-latest .

# Opt-in DeepStream variant (ADR-0010 amendment "Single-Dockerfile
# consolidation"). DeepStream is proprietary + EULA-restricted; this
# target refuses to run unless the user has downloaded the SDK tarball
# locally (signalling EULA acceptance). The tarball lives at
# `docker/inference/deepstream/deepstream_sdk_v9.0.0_x86_64.tbz2`
# (gitignored) and is loaded via a BuildKit named context so it stays
# OUT of the default build context for non-DS builds.
# The resulting image is NEVER pushed to GHCR — it's for local /
# private-registry use only. Read `docker/inference/README.md` first.
docker-build-x86-deepstream:
    @test -f docker/inference/deepstream/deepstream_sdk_v9.0.0_x86_64.tbz2 || ( \
        echo ""; \
        echo "ERROR: deepstream_sdk_v9.0.0_x86_64.tbz2 not found in docker/inference/deepstream/."; \
        echo ""; \
        echo "Before building this image you must:"; \
        echo "  1. Read docker/inference/README.md (EULA implications)."; \
        echo "  2. Read the NVIDIA DeepStream EULA at"; \
        echo "     https://developer.download.nvidia.com/assets/Deepstream/LicenseAgreement-NGC.pdf"; \
        echo "  3. Download the SDK from https://developer.nvidia.com/deepstream-getting-started"; \
        echo "     (~1.5 GB; requires NGC account)."; \
        echo "  4. Move the tarball into docker/inference/deepstream/."; \
        echo ""; \
        exit 1 \
    )
    docker buildx build -f docker/inference/Dockerfile.x86 \
        --build-arg WITH_DEEPSTREAM_STAGE=on \
        --build-context ds=docker/inference/deepstream/ \
        -t openral:x86-deepstream-latest .

# Live ROS-tee round-trip inside the consolidated x86 image (which
# always carries ROS 2 Jazzy after the consolidation): starts the
# reader's publisher, spins a real rclpy subscriber, asserts a real
# sensor_msgs/Image arrives. Exits non-zero on failure.
docker-smoke-x86-ros-tee: docker-build-x86
    docker run --rm --gpus all \
        --entrypoint /entrypoint.sh \
        -v "$(pwd)/docker/inference/smoke_ros_tee.py:/workspace/smoke_ros_tee.py:ro" \
        openral:x86-latest \
        python /workspace/smoke_ros_tee.py

# Live perception-tee round-trip inside the x86 image (ADR-0018 F6):
# starts a PerceptionEventPublisher fed by a videotestsrc pattern=ball
# event leg, spins a real rclpy subscriber on
# /openral/perception/motion, asserts at least one PromptStamped
# arrives whose metadata_json decodes through PerceptionEventMetadata
# into a MotionMetadata. Exits non-zero on failure.
#
# The colcon-built openral_msgs IDL is now baked into the image
# (`/workspace/install/`) by the Dockerfile builder stage, so the
# previous "host must run `just ros2-build` first" guard is gone. The
# entrypoint sources the local install overlay automatically.
docker-smoke-x86-perception-tee: docker-build-x86
    docker run --rm --gpus all \
        -v "$(pwd)/docker/inference/smoke_perception_tee.py:/workspace/smoke_perception_tee.py:ro" \
        --entrypoint /entrypoint.sh \
        openral:x86-latest \
        python /workspace/smoke_perception_tee.py

# Live reasoner + prompt-router round-trip inside the x86 image
# (ADR-0018 F4 + F10): spins a real PromptRouterNode + ReasonerNode in
# the same executor, publishes one operator prompt on
# /openral/prompt_in/cli, asserts the reasoner's EmitPromptTool reply
# lands on /openral/prompt with frame_id=openral_reasoner AND a
# traceparent stamped into metadata_json (the OTel span integration).
# Uses a FakeToolUseClient mirrored inline — no live LLM, no API key.
# Exits non-zero on failure.
docker-smoke-x86-reasoner: docker-build-x86
    docker run --rm --gpus all \
        -v "$(pwd)/docker/inference/smoke_reasoner.py:/workspace/smoke_reasoner.py:ro" \
        --entrypoint /entrypoint.sh \
        openral:x86-latest \
        python /workspace/smoke_reasoner.py

# Verify the C++ safety kernel binary is present + executable inside
# the image. Cheap smoke that catches a regression in the colcon build
# step (e.g. opentelemetry_cpp_vendor failing to fetch the upstream
# tarball) before the next on-call sees a missing /workspace/install/
# openral_safety_kernel/bin/safety_kernel.
docker-smoke-x86-safety-kernel: docker-build-x86
    docker run --rm --entrypoint /entrypoint.sh openral:x86-latest \
        bash -c \
        'set -e; \
         test -x /workspace/install/lib/openral_safety_kernel/safety_kernel_node \
           || { echo "MISSING: safety_kernel_node binary not baked into image"; \
                find /workspace/install -name "safety_kernel*" 2>/dev/null; \
                exit 1; }; \
         echo "OK: safety_kernel_node present at $(readlink -f /workspace/install/lib/openral_safety_kernel/safety_kernel_node)"; \
         python -c "import openral_msgs.msg, openral_msgs.action; print(\"OK: openral_msgs IDL importable\")"; \
         openral doctor | head -20'

# All Python unit tests (no ROS 2 required).
#
# GStreamer tests run in a separate pytest invocation because PyGObject's
# system GLib does not coexist in the same process as torch's bundled GLib
# (segfault on x86 dev hosts where ``gi`` resolves to /usr/lib/python3/
# dist-packages). The Docker images (Commit #7) install gi from apt under a
# matching GLib so both can co-load. See ADR-0010 amendment for context.
#
# The second invocation only runs when ``gi`` (PyGObject) is importable —
# pytest 8 exits 4 (USAGE_ERROR) on ``-q + no collectors`` for the
# module-level ``importorskip`` files, which would otherwise wedge
# ``just test`` on hosts that legitimately don't have the optional
# extra installed.
#
# `-p no:launch_testing -p no:launch_ros` are mandatory whenever a dev
# shell has ROS 2 Jazzy sourced (a common dev setup since `just ros2-build`
# and `just test-integration` require it). The launch_testing pytest plugin
# auto-loads from /opt/ros/jazzy/lib/python3.12/site-packages and overrides
# `pytest_pycollect_makemodule` to pre-import every test file at collection
# time looking for `generate_test_description`. That pre-import fights with
# the workspace's lazy imports / conftest setup and silently aborts unit-test
# collection (0 tests gathered, exit code 5). Disabling the two ROS plugins
# is safe here because unit tests never use `launch_testing`.
test:
    uv run pytest tests/unit/ -q -p no:launch_testing -p no:launch_ros --ignore=tests/unit/test_gstreamer_pipeline.py --ignore=tests/unit/test_gstreamer_sensor_reader.py --ignore=tests/unit/test_gstreamer_ros_tee.py --ignore=tests/unit/test_gstreamer_perception_tee.py
    @if uv run python -c "import gi" >/dev/null 2>&1; then \
        uv run pytest tests/unit/test_gstreamer_pipeline.py tests/unit/test_gstreamer_sensor_reader.py tests/unit/test_gstreamer_ros_tee.py tests/unit/test_gstreamer_perception_tee.py -q -p no:launch_testing -p no:launch_ros; \
    else \
        echo "skipping gstreamer tests: PyGObject not installed (pip install openral-runner[gstreamer] or use docker/inference/Dockerfile)"; \
    fi

# Run docstring examples on the curated set of packages (CLAUDE.md §5.4).
# Update tests/unit/test_doctest_runner.py::DOCTEST_TARGETS when adding more.
# `-p no:launch_testing -p no:launch_ros`: same ROS-env workaround as
# `just test` — the launch_testing pytest plugin silently aborts
# collection when ROS 2 Jazzy is sourced.
test-doctest:
    uv run pytest --doctest-modules -q -p no:launch_testing -p no:launch_ros \
        python/core/src/openral_core \
        python/cli/src/openral_cli \
        python/sensors/src/openral_sensors \
        python/world_state/src/openral_world_state \
        python/hal/src/openral_hal/protocol.py \
        python/hal/src/openral_hal/sim_transport.py \
        python/hal/src/openral_hal/_real_description.py \
        python/hal/src/openral_hal/franka_panda.py \
        python/hal/src/openral_hal/franka_panda_real.py \
        python/hal/src/openral_hal/sawyer_real.py \
        python/hal/src/openral_hal/aloha.py \
        python/hal/src/openral_hal/ur.py \
        python/hal/src/openral_hal/ur_real.py \
        python/hal/src/openral_hal/so100_sim.py \
        python/hal/src/openral_hal/lifecycle.py \
        python/hal/src/openral_hal/so100_follower.py \
        python/hal/src/openral_hal/ros_control.py \
        python/rskill/src/openral_rskill/base.py \
        python/rskill/src/openral_rskill/engine_cache.py \
        python/rskill/src/openral_rskill/loader.py \
        python/rskill/src/openral_rskill/quantization.py \
        python/rskill/src/openral_rskill/runtime.py \
        python/rskill/src/openral_rskill/runtime_onnx.py \
        python/rskill/src/openral_rskill/runtime_pytorch.py \
        python/rskill/src/openral_rskill/smolvla.py \
        python/reasoner/src/openral_reasoner \
        python/wam/src/openral_wam

# Sim integration tests: real HF weights + GPU + simulated robots/envs (slow, opt-in)
# `-p no:launch_testing -p no:launch_ros`: same ROS-env workaround as
# `just test` — the launch_testing pytest plugin silently aborts
# collection when ROS 2 Jazzy is sourced.
test-sim:
    uv run pytest tests/sim/ -m sim -v -p no:launch_testing -p no:launch_ros

# HAL digital-twin sweep — every shipping HAL exercised both in isolation
# (tests/sim/test_<robot>_hal_mujoco.py) and end-to-end through the
# production HardwareRunner (tests/sim/test_all_hals_via_runner.py).
# This is the "are the HAL contracts honoured by every robot, and does
# the inference-runner wiring work for each" sweep — the strongest
# pre-hardware signal we ship.
#
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1: when ROS 2 Jazzy is sourced the
# ament-* and launch-testing-* pytest plugins auto-load from system
# Python and conflate module-level pytest.skip reasons across files,
# silently dropping every HAL twin test under a single misleading
# "ERROR: found no collectors" line.  See tests/sim/conftest.py.
hal-twin-sweep:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
        tests/sim/ -m sim \
        -k "hal_mujoco or all_hals_via_runner" \
        -v --no-header

# ROS 2-gated integration tests (requires: source /opt/ros/jazzy/setup.bash)
test-integration:
    PYTHONPATH="$(pwd)/packages/world_state:${PYTHONPATH}" uv run pytest tests/integration/ -v --tb=short

# Subset by keyword
# `-p no:launch_testing -p no:launch_ros`: same ROS-env workaround as
# `just test` — the launch_testing pytest plugin silently aborts
# collection when ROS 2 Jazzy is sourced.
test-k k:
    uv run pytest -q -p no:launch_testing -p no:launch_ros -k "{{k}}"

# Selective testing — print the pytest targets a diff actually needs.
# See docs/contributing/selective-testing.md. Defaults to origin/master..HEAD.
test-changed base="origin/master" head="HEAD":
    uv run python tools/select_tests.py --base "{{base}}" --head "{{head}}"

# Selective testing — actually run only the affected tests for the diff.
# Falls back to the full unit suite on a blast-radius change (full_run=true).
test-changed-run base="origin/master" head="HEAD":
    #!/usr/bin/env bash
    set -euo pipefail
    plan=$(uv run python tools/select_tests.py --base "{{base}}" --head "{{head}}" --github-output)
    # Fork-in-threaded-process crashers (issue #24) run in their own process.
    isolated=$(echo "$plan" | python3 -c "import sys,json; print(json.load(sys.stdin)['isolated_targets'])" 2>/dev/null || true)
    ignore=""
    for t in $isolated; do ignore="$ignore --ignore=$t"; done
    rc=0
    if echo "$plan" | grep -q '"full_run": "true"'; then
        echo "Blast-radius change → running full unit suite"
        uv run pytest tests/unit/ $ignore -q -p no:launch_testing -p no:launch_ros || rc=1
    else
        targets=$(echo "$plan" | python3 -c "import sys,json; print(json.load(sys.stdin)['targets'])" 2>/dev/null || true)
        if [ -z "${targets// }" ] && [ -z "${isolated// }" ]; then
            echo "No code paths affected by this diff — nothing to run."
        elif [ -n "${targets// }" ]; then
            echo "Running selected targets: $targets"
            uv run pytest $targets $ignore -q -p no:launch_testing -p no:launch_ros || rc=1
        fi
    fi
    for t in $isolated; do
        echo "Running isolated target (own process): $t"
        uv run pytest "$t" -q -p no:launch_testing -p no:launch_ros || rc=1
    done
    exit $rc

# Audit the test suite (dead / shadowed / duplicate / no-assertion) and refresh
# docs/contributing/test-audit.md. Read-only — never deletes tests.
test-audit:
    uv run python tools/audit_tests.py --write-report

# Print ROS 2 env info and source reminder
ros2-env:
    @echo "ROS_DISTRO=${ROS_DISTRO:-<not set>}"
    @echo "ros2 binary: $(which ros2 2>/dev/null || echo 'not found')"
    @echo "To source: source /opt/ros/jazzy/setup.bash"

# ROS 2 build (selected packages only).
#
# Points CMake at the workspace's uv-managed `.venv/bin/python` (the same
# interpreter that imports the colcon-installed Python lifecycle nodes at
# runtime). The previous incarnation of this recipe pinned
# `Python3_EXECUTABLE=/usr/bin/python3` to satisfy ament's configure-time
# imports of `catkin_pkg` / `em` (provided by apt's `python3-catkin-pkg`
# / `python3-empy` on the *system* Python), but that broke every test
# that imports `structlog` or any `openral_*` package — uv installs those
# via editable `.pth` files inside `.venv/lib/python3.12/site-packages/`,
# and PYTHONPATH alone cannot bridge them into the system Python because
# `.pth` files are only processed for directories that `site.py` knows as
# site-dirs, not for arbitrary PYTHONPATH entries. Routing through the
# venv interpreter sidesteps that asymmetry entirely. `catkin-pkg` and
# `empy<4` are pinned in the workspace dev deps (`pyproject.toml`) so
# they land inside `.venv/` on `uv sync --all-packages` and ament's
# configure step finds them on the venv's own `sys.path` — no
# sitecustomize / PYTHONPATH bridge required. Matches `.github/workflows/
# test-ros2.yml`, which also activates the venv before colcon.
#
# The list mirrors every colcon package currently on disk under packages/ and
# cpp/ that is wired into the ADR-0018 ROS 2 graph (F1 skill_runner,
# F4 reasoner, F5 safety, F8 world_state, F10 prompt_router, plus the
# C++ safety kernel from PR #138 and the HAL nodes). When a new ROS 2
# package lands under packages/ or cpp/, add it here so `just ros2-build`
# stays the single-command rebuild for the whole graph.
ros2-build:
    colcon build --merge-install --symlink-install \
        --base-paths packages cpp \
        --packages-select openral_msgs \
                          opentelemetry_cpp_vendor \
                          openral_hal_so100 \
                          openral_hal_panda_mobile \
                          openral_hal_openarm \
                          openral_hal_franka \
                          openral_hal_ur5e \
                          openral_hal_ur10e \
                          openral_hal_aloha \
                          openral_hal_g1 \
                          openral_hal_h1 \
                          openral_hal_rizon4 \
                          openral_world_state \
                          openral_reasoner_ros \
                          openral_prompt_router \
                          openral_safety \
                          openral_safety_watchdog \
                          openral_safety_kernel \
                          openral_human_estop \
                          openral_rskill_ros \
                          openral_slam_bringup \
                          openral_nav2_bringup \
                          openral_octomap_bridge \
                          openral_perception_ros \
        --cmake-args -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
                     -DCMAKE_C_COMPILER=/usr/bin/gcc \
                     -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
                     -DCMAKE_POLICY_VERSION_MINIMUM=3.10
# CMAKE_POLICY_VERSION_MINIMUM=3.10 — ROS Jazzy's bundled gtest_vendor
# CMakeLists declares ``cmake_minimum_required(VERSION 3.5)``, which
# CMake >= 3.27 flags with a "compatibility with CMake < 3.10 will be
# removed" deprecation warning. That noise leaks through colcon as
# stderr on every package that pulls gtest_vendor (every ament_cmake
# package with tests, basically). We can't patch /opt/ros/jazzy from
# our tree, so we bump the effective policy floor at the colcon
# boundary — silences the deprecation without disabling other warnings.
# The same flag is also set inside opentelemetry_cpp_vendor's own
# ExternalProject_Add CMAKE_ARGS for the upstream otel-cpp 1.16.1 tree.

# ROS 2 test — scope must match `ros2-build`'s `--packages-select` set,
# otherwise colcon discovers (and tries to test) packages it never built
# and reports them as failures with "Has this package been built before?".
#
# The Python interpreter is already baked into each package's
# CTestTestfile by the `ros2-build` step above (`Python3_EXECUTABLE` ->
# `.venv/bin/python`), so test invocations resolve `structlog`,
# `openral_*`, and the ament/ROS bits through the venv automatically.
ros2-test:
    colcon test --merge-install \
        --packages-select openral_msgs \
                          opentelemetry_cpp_vendor \
                          openral_hal_so100 \
                          openral_hal_panda_mobile \
                          openral_hal_openarm \
                          openral_hal_franka \
                          openral_hal_ur5e \
                          openral_hal_ur10e \
                          openral_hal_aloha \
                          openral_hal_g1 \
                          openral_hal_h1 \
                          openral_hal_rizon4 \
                          openral_world_state \
                          openral_reasoner_ros \
                          openral_prompt_router \
                          openral_safety \
                          openral_safety_watchdog \
                          openral_safety_kernel \
                          openral_human_estop \
                          openral_rskill_ros \
                          openral_slam_bringup \
                          openral_nav2_bringup \
                          openral_octomap_bridge \
                          openral_perception_ros
    colcon test-result --verbose

# Run a SimEnvironment YAML config end-to-end via the eval registry/runner.
# This is the canonical config-driven entry point — see ADR-0002.
sim-eval config *args:
    uv run openral sim run --config {{config}} {{args}}

# hf-libero 0.1.3 ships both .dist-info AND .egg-info metadata in its wheel,
# so uv reports it twice and refuses to switch groups. Strip the duplicate
# egg-info before each libero recipe runs.
_strip-hf-libero-egg:
    @find .venv -maxdepth 5 -name 'hf_libero-*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# LIBERO persists ~/.libero/config.yaml the first time the package is
# imported, pinning absolute paths to the libero data dirs. Switching
# venv / clone / workspace path leaves the file stale and the next sim
# run crashes inside lerobot.envs.libero with a confusing
# FileNotFoundError on init_files/<task>.pruned_init. The helper
# rewrites the config only when stale; idempotent.
_ensure-libero-config:
    @uv run --group libero python tools/fix_libero_config.py 2>/dev/null || true

# Real LIBERO sim eval (downloads lerobot/smolvla_libero, requires MUJOCO_GL=osmesa or egl).
# Writes example_videos/libero_spatial.mp4 by default.
# Defaults to EGL — osmesa fails with "Default framebuffer is not complete, error 0x0"
# on the LIBERO MJCF render contexts on most desktop Linux setups.
#
# `--all-packages` is required: without it, `uv run --group libero` evicts
# the workspace's `openral` console script (and other openral_* packages) when
# uv switches dependency groups, breaking the next run with
# "Failed to spawn: `openral`". `--rskill` is required by ADR-0009 — the YAML
# config carries the scene/task/robot tuple while the rSkill manifest
# carries the policy + action contract.
sim-libero *args: _strip-hf-libero-egg _ensure-libero-config
    MUJOCO_GL=egl uv run --all-packages --group libero openral sim run --config scenes/sim/libero_spatial.yaml --rskill rskill://rskills/smolvla-libero --save-video example_videos {{args}}

# LIBERO sim eval with xVLA / Florence-2 (SimScene tier — smoke demo, not a paper-comparable benchmark).
sim-xvla-libero *args: _strip-hf-libero-egg _ensure-libero-config
    MUJOCO_GL=egl uv run --all-packages --group libero openral sim run --config scenes/sim/libero_spatial.yaml --rskill rskill://rskills/xvla-libero --save-video example_videos {{args}}

# LIBERO sim eval with π0.5 (SimScene tier; requires ≥8 GB VRAM; weights are non-commercial).
sim-pi05-libero *args: _strip-hf-libero-egg _ensure-libero-config
    MUJOCO_GL=egl uv run --all-packages --group libero openral sim run --config scenes/sim/libero_spatial.yaml --rskill rskill://rskills/pi05-libero-nf4 --save-video example_videos {{args}}

# MetaWorld push-v2 with SmolVLA — 1-episode demo via the BenchmarkScene tier
# (no SimScene sibling for MetaWorld today). `--no-update-manifest` keeps the
# rSkill's recorded benchmark numbers untouched on a demo run; lift the flag
# (and bump `--n-episodes`) when you want a paper-comparable claim.
# Uses EGL (not osmesa) because osmesa hits "Default framebuffer is not complete, error 0x0"
# on Sawyer's MetaWorld scenes; EGL renders offscreen against the GPU's framebuffer object.
# Requires the metaworld benchmark package to be installed once:
#   uv run pip install metaworld==3.0.0 --no-deps
sim-metaworld *args:
    MUJOCO_GL=egl uv run --all-packages --group metaworld openral benchmark scene --config scenes/benchmark/metaworld_push.yaml --rskill rskill://rskills/smolvla-metaworld --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# gym-aloha bimanual cube-transfer with ACT — BenchmarkScene tier demo (1 episode).
# Uses EGL (not osmesa) for the same reason as sim-metaworld: osmesa fails with
# "Default framebuffer is not complete, error 0x0" on the aloha MJCF render context.
sim-act-aloha *args:
    MUJOCO_GL=egl uv run --all-packages --group sim openral benchmark scene --config scenes/benchmark/aloha_transfer_cube.yaml --rskill rskill://rskills/act-aloha --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# gym-pusht with Diffusion Policy (pymunk 2-D rigid body) — BenchmarkScene tier demo (1 episode).
sim-diffusion-pusht *args:
    uv run --all-packages --group sim openral benchmark scene --config scenes/benchmark/pusht.yaml --rskill rskill://rskills/diffusion-pusht --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# ManiSkill3 PickCube-v1 — 1-episode plumbing rollout against the adapter
# (no MS3-specific rSkill ships in-tree yet; pass --rskill rskill://rskills/<id>
# via {{args}} to swap in your own). Uses the BenchmarkScene tier so the
# scene/task/seed plumbing matches what tests/sim covers end-to-end.
sim-maniskill3 *args:
    uv run --all-packages --group maniskill3 openral benchmark scene --config scenes/benchmark/maniskill_pick_cube.yaml --rskill rskill://rskills/smolvla-maniskill-franka --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# SimplerEnv WidowX carrot-on-plate — BenchmarkScene tier demo (1 episode).
# Pairs with the rldx1-ft-simpler-widowx-nf4 rSkill (heavy out-of-process
# sidecar; bootstraps lazily on first run).
sim-simpler-widowx *args:
    uv run --all-packages --group simpler-env openral benchmark scene --config scenes/benchmark/widowx_carrot_on_plate.yaml --rskill rskill://rskills/rldx1-ft-simpler-widowx-nf4 --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# Custom example — ACT (insertion checkpoint) x gym-aloha AlohaInsertion-v0.
# BenchmarkScene tier demo (1 episode); pairs with the rskills/act-aloha-insertion
# rSkill. See scenes/benchmark/aloha_insertion.yaml for the canonical eval shape.
sim-custom *args:
    MUJOCO_GL=egl uv run --all-packages --group sim openral benchmark scene --config scenes/benchmark/aloha_insertion.yaml --rskill rskill://rskills/act-aloha-insertion --no-update-manifest --n-episodes 1 --save-dir example_videos {{args}}

# LIBERO sim eval with ACT (SimScene tier — per-policy comparison against
# smolvla/xvla/pi05 on the same suite).
sim-act-libero *args: _strip-hf-libero-egg _ensure-libero-config
    MUJOCO_GL=egl uv run --all-packages --group libero openral sim run --config scenes/sim/libero_spatial.yaml --rskill rskill://rskills/act-libero --save-video example_videos {{args}}

# RoboCasa PickPlaceCounterToCabinet with π0.5 (SimScene tier; NF4 already; ~5 GB VRAM).
sim-pi05-robocasa *args:
    MUJOCO_GL=egl uv run --all-packages --group robocasa openral sim run --config scenes/sim/robocasa_pnp.yaml --rskill rskill://rskills/pi05-robocasa365-human300-nf4 --save-video example_videos {{args}}

# Full end-to-end audit of every YAML under scenes/.
# 1 episode per config, real GPU rollout (CLAUDE.md §1.11–§1.12), JSON report
# written to outputs/audit_sim_configs.json. Per-config timeout 10 min.
# Pass YAML stems as args to narrow scope:
#   just sim-audit libero_spatial robocasa_pnp
sim-audit *args:
    uv run python tools/audit_sim_configs.py {{args}}

# Hardware-in-loop (requires connected robot + permissions)
# HIL test driver. When no hardware is attached, the per-file module-level
# `pytestmark = pytest.mark.skipif(...)` fires before any test is collected,
# which pytest reports as exit 4 ("no collectors") rather than 5 ("no tests
# collected"). We normalise that into a clean skip-with-message so dev boxes
# without robots see "SKIPPED (no hardware)" instead of a recipe failure.
hil robot:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run pytest -q tests/hil/test_{{robot}}.py
    status=$?
    if [[ $status -eq 4 || $status -eq 5 ]]; then
        echo "SKIPPED: no hardware connected for {{robot}} (set the appropriate env var or attach the robot to run)."
        exit 0
    fi
    exit $status

# Docs serve
docs:
    uv run mkdocs serve

# Docs build (CI parity)
docs-build:
    uv sync --all-packages
    uv run mkdocs build --strict

# Schema export (CI compares)
schema-export:
    uv run python tools/schema_export.py

# Scaffold a new local rSkill from rskills/template/ (interactive prompts)
skill-new id:
    uv run openral skill new {{ id }}

# ADR-0020 — build just the C++ safety kernel (and its dependencies).
# Requires a sourced ROS 2 environment and the workspace venv on $PATH so
# rosidl_adapter picks up empy 3.x.
safety-kernel-build:
    colcon build --merge-install --base-paths packages cpp --packages-select \
        openral_msgs opentelemetry_cpp_vendor openral_safety_kernel \
        --cmake-args -DBUILD_TESTING=ON \
                     -DPython3_EXECUTABLE=$(which python)

# ADR-0020 — run the C++ kernel's gtest + lifecycle test suite. CI parity
# with `colcon test`; linter failures (cpplint, flake8, pep257,
# uncrustify, xmllint) are reported but do not gate the recipe — the
# functional gtest binaries do.
safety-kernel-test:
    colcon test --merge-install --packages-select openral_safety_kernel \
        --event-handlers console_direct+
    colcon test-result --verbose --test-result-base \
        build/openral_safety_kernel/test_results/openral_safety_kernel

# ADR-0020 — clang-format + clang-tidy. Requires both binaries on $PATH.
safety-kernel-lint:
    #!/usr/bin/env bash
    set -euo pipefail
    find cpp/openral_safety_kernel/include cpp/openral_safety_kernel/src cpp/openral_safety_kernel/test \
        \( -name '*.cpp' -o -name '*.hpp' \) -print0 \
        | xargs -0 clang-format --dry-run --Werror -style=file
    if command -v clang-tidy >/dev/null 2>&1; then
        find cpp/openral_safety_kernel/src \
            -name '*.cpp' -print0 \
            | xargs -0 -I{} clang-tidy -p build/openral_safety_kernel \
                --config-file=cpp/openral_safety_kernel/.clang-tidy {}
    else
        echo "clang-tidy not on PATH — skipping (CI parity gate)"
    fi

# ADR-0020 — run safety-kernel sim tests with real GPU + ROS. Skips
# cleanly when the requested config is absent or no GPU is detected.
sim-safety config *args:
    MUJOCO_GL=egl uv run --group sim openral sim run \
        --config scenes/{{config}} \
        {{args}}

# ADR-0020 — HIL safety tests for a robot (today: so100 only). Mirrors
# the existing `hil` recipe's no-hardware skip semantics.
hil-safety robot:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run pytest -q tests/hil/safety/test_{{robot}}_*.py
    status=$?
    if [[ $status -eq 4 || $status -eq 5 ]]; then
        echo "SKIPPED: no hardware connected for {{robot}}."
        exit 0
    fi
    exit $status

# Clean
clean:
    rm -rf build install log .pytest_cache .ruff_cache .mypy_cache
    find . -name __pycache__ -type d -exec rm -rf {} +
