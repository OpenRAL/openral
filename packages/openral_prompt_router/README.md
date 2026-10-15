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
`/openral/prompt`; sources never publish there directly.

## v1 adapters

| Source | Priority | Adapter |
|---|---|---|
| `cli` | 100 (human) | `openral prompt "do X"` — see [`openral_cli.prompt`](../../python/cli/src/openral_cli/prompt.py) |
| `dashboard` | 100 (human) | Future WebSocket adapter (post-v1) |
| `auto` | 10 (machine) | `EmitPromptTool` self-cascades from the F4 reasoner |

Per-source allowlist comes from the constructor's `sources` dict; the
deployment YAML may restrict the set (per capability review §3.F10,
the "per-source allowlist in deployment YAML").

## QoS

`/openral/prompt` and every `/openral/prompt_in/<source>` use
`RELIABLE + VOLATILE + KEEP_LAST=10`. A saturated subscriber surfaces as a
structlog warning, never a silently dropped message.

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
