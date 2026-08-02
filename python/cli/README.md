# openral-cli

OpenRAL command-line tool

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
eight-layer model in the project docs.

- **Docs:** https://openral.github.io/openral/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep at `0.1.x` until the first
> public release.

## BEHAVIOR Challenge

Serve an R1Pro-compatible rSkill through the official BEHAVIOR WebSocket
protocol, then run OmniGibson's evaluator in its own environment:

```bash
openral behavior serve \
  --rskill rskills/gr00t-n17-b1k-turning-on-radio \
  --task turning_on_radio

conda run -n behavior python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 \
  --output-dir outputs/openral --write-video
```

The bridge defaults to the official R1Pro contract: three RGB cameras, 61-D
proprioception, and a 23-D action.

The same contract can run through the full deploy graph:

```bash
openral deploy sim --config scenes/deploy/behavior_r1pro.yaml \
  --initial-task "turn on the radio"
```
