# Design decisions

OpenRAL records architecture and design decisions as Architecture Decision Records (ADRs). The log lives in the private [`OpenRAL/management`](https://github.com/OpenRAL/management) repo, under `adr/`, not in this repo.

Code, docstrings, tests and docs here cite decisions by bare `ADR-NNNN` ids. They are deliberately not links: the record is private, and a dead link is worse than a bare id. The surrounding prose stands on its own, so a reader without access can follow what the behavior is.

- The number is a handle, not a coordinate. The log is append-only but not gap-free; do not infer order, date or dependency from neighbouring numbers.
- Without access, ask for the rationale behind a decision by opening an issue here that quotes the id.
- Moving a responsibility between the seven architecture layers needs a decision in the private log before the code ([CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md) section 3).

## Licensing and commercial boundary

These decisions define the public/private boundary. They are summarized here so the posture stays visible.

| Decision | What it means |
|---|---|
| **Public repo is uniformly Apache-2.0** | Every package ships under one license: no source-available tier, no per-package drift. Copy-left dependencies are rejected without TSC review. |
| **Commercial capabilities live in `OpenRAL/openral-pro`** (private) | TensorRT/NVMM zero-copy runtime, WAM implementations, fleet/cloud dispatch, premium rSkills. This repo keeps the protocols and entry-point seams, not the implementations. |
| **Safety kernel stays open; refusal-only checks stay open** (ADR-0121) | A rule that relaxes a verdict (lets a hand touch a cell, ends a payload binding) may move to Pro, only behind the kernel's open, fail-closed, sim-only exemption provider (`cpp/openral_safety_kernel/include/openral_safety_kernel/exemption_provider.hpp`). The kernel enforces the provider's bounds itself. |
| **Open tracking mode uses declarations only** (ADR-0122) | The open tracker (`object_tracking: tracking`) only adds checked geometry, such as a carried payload. Every contact allowance there comes from a scene declaration (`search_box`, a place zone, `workspace_box`). Adaptive, perception-driven allowances are `object_tracking: on`, which requires an exemption provider (Pro). A release is the kernel's in every mode. |
| **Weights of 8 bits or fewer need Pro** (ADR-0123) | Open runs fp32, fp16, bf16. int8, fp8, int4/NF4 and nvfp4 need Pro, whether quantized at load time, pre-quantized or quantized upstream. Quantized rSkills stay in the catalog and refuse to load without Pro, with an error naming it. The gate is `python/rskill/src/openral_rskill/quantization_gate.py` (`openral.pro_modules: quantization`). A 4-bit reasoner LLM served by an operator-run server such as Ollama stays open. |
| **Dashboard sign-in check is open** (ADR-0129) | Token validation, roles, the remote listener and the action audit live in `openral-observability`. Every Pro limit (robot and teammate counts, fleet mode, compliance history) is enforced by openral.com when it issues tunnels and tokens, not on the robot. |
| **Robot console ships as a prebuilt bundle** (ADR-0131) | The bundle is Apache-2.0; its source is private. The dashboard downloads the build pinned by sha256 (`ui_assets.ROBOT_UI`). The server, robot API, manifest schema (`openral_core.robot_ui`) and fallback page with its E-STOP button stay here; `OPENRAL_DASHBOARD_UI_DIR` serves any other UI. See the [dashboard guide](quickstart/dashboard.md#the-page). |
| **Model weights keep their upstream license** | E.g. NVIDIA GR00T checkpoints. This is compliance for models OpenRAL does not own, not a statement about OpenRAL's code. |

Context, alternatives and consequences are in the private decision log.
