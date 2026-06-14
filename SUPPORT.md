# Getting help with OpenRAL

Thanks for using OpenRAL! Here's where to go depending on what you need.

## I have a question or want to discuss an idea

- **[Discord](https://discord.gg/ZdNyUT4V5)** — the fastest way to get help, ask
  "how do I…?" questions, share what you're building, or float a design idea
  before opening an issue.
- **[Documentation](https://docs.openral.dev)** — quick start, architecture,
  tutorials, and the API reference. Start with the
  [quick start](https://docs.openral.dev) and the
  [architecture overview](docs/architecture/overview.md).
- **General enquiries** that don't fit Discord or an issue: hello@openral.dev.

## I think I found a bug

Open a [bug report](https://github.com/OpenRAL/openral/issues/new?template=bug.yml).
Please run `openral doctor --json` first and paste the output — it tells us your
OS, Python, ROS 2 distro, and GPU in one shot.

## I want to request a feature

Open a [feature request](https://github.com/OpenRAL/openral/issues/new?template=feature.yml).
For anything that crosses a layer boundary, expect to be asked for an ADR
(see [CLAUDE.md](CLAUDE.md) §3 and `docs/adr/`).

## I found a security vulnerability or a safety defect

**Do not open a public issue.** Follow [SECURITY.md](SECURITY.md):
use [private vulnerability reporting](https://github.com/OpenRAL/openral/security/advisories/new)
or email security@openral.dev. Physical-safety defects (E-stop bypass,
actuation-path bugs, safety-kernel issues) are treated as highest priority —
you can also reach the Safety Working Group directly at safety@openral.dev.

## I want to contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the engineering playbook in
[CLAUDE.md](CLAUDE.md) before opening a PR.

---

> Note: GitHub issues are for **confirmed bugs and concrete feature requests**.
> Usage questions and open-ended discussion belong on Discord so the issue
> tracker stays a clean signal of work to be done.
