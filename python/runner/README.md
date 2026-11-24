# openral-runner

OpenRAL inference runner: closes the WorldState → rSkill → HAL loop at a fixed
rate. Hosts the `InferenceRunner` protocol and rate-limited base class, the
hardware-side sensor readers and the deploy runner; `openral_sim` is its
simulation counterpart.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.
