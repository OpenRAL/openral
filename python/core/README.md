# openral-core

OpenRAL Pydantic schemas and exceptions: the normative contract for the layer
interfaces. Defines `RobotDescription`, `RSkillManifest`, `DeployScene`,
`WorldState`, the `ROSError` hierarchy and the manifest loaders, with
geometry, kinematics and time-base helpers shared by every other package.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
seven-layer model in the project docs.

- **Docs:** https://docs.openral.com/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep with the workspace SemVer
> version; the bump is computed from Conventional Commits by release-please.
