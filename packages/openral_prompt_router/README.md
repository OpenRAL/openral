# `openral_prompt_router`

Lifecycle node that fans in operator prompts from any source into a
normalised `openral_msgs/PromptStamped` stream
on `/openral/prompt`.

## Topology

```
/openral/prompt_in/cli        \
/openral/prompt_in/dashboard   } --[prompt_router_node]--> /openral/prompt
/openral/prompt_in/auto       /
```

Each source listens on `/openral/prompt_in/<source>`; the router stamps
`{"source": "<source>", "priority": <int>}` into `metadata_json` and
republishes on `/openral/prompt`. The F4 reasoner consumes only
`/openral/prompt`. Operator sources publish on their `prompt_in` topic; the
reasoner's own cascade re-prompts go straight to `/openral/prompt`.

## v1 adapters

| Source | Priority | Adapter |
|---|---|---|
| `cli` | 100 (human) | `openral prompt "do X"` — see [`openral_cli.prompt`](../../python/cli/src/openral_cli/prompt.py) |
| `dashboard` | 100 (human) | Dashboard prompt input — shells out to `openral prompt --topic /openral/prompt_in/dashboard` |
| `auto` | 10 (machine) | Machine-source slot; no in-tree producer publishes on it by default (an `EmitPromptTool` call whose `target_topic` names it would) |

Per-source allowlist comes from the constructor's `sources` dict (default
`DEFAULT_SOURCES`, the three above); it is not a ROS parameter.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `startup_prompt` | `""` | Published once at `on_activate` as a `cli`-priority prompt (after waiting up to 30 s for the reasoner's subscriber); `openral deploy sim` passes the scene's goals here. Empty = idle until an operator prompt. |

## QoS

`/openral/prompt` and every `/openral/prompt_in/<source>` use
`RELIABLE + VOLATILE + KEEP_LAST=10`.

## Synopsis

```bash
just ros2-build      # builds openral_msgs + openral_prompt_router
source install/setup.bash

ros2 run openral_prompt_router prompt_router_node
ros2 lifecycle set /openral_prompt_router configure
ros2 lifecycle set /openral_prompt_router activate

# CLI adapter — publishes on /openral/prompt_in/cli; the router fans
# out to /openral/prompt with source=cli + priority=100.
openral prompt "pick the red cube"
```

## See also

- [`packages/openral_reasoner_ros`](../openral_reasoner_ros/) — F4 consumer.
- [`openral_cli.prompt`](../../python/cli/src/openral_cli/prompt.py) — `openral prompt` CLI entry point.
