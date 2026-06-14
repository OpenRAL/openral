# `openral_prompt_router`

Single ROS 2 lifecycle node that fans in operator prompts from any
external source into a normalised `openral_msgs/PromptStamped` stream
on `/openral/prompt` (ADR-0018 F10).

## Topology

```
/openral/prompt_in/cli        \
/openral/prompt_in/dashboard   } --[prompt_router_node]--> /openral/prompt
/openral/prompt_in/auto       /
```

Each registered source listens on `/openral/prompt_in/<source>` and the
router republishes the message onto `/openral/prompt` after stamping
`{"source": "<source>", "priority": <int>}` onto the
`metadata_json`. The F4 reasoner consumes `/openral/prompt`
exclusively — sources never publish there directly.

## v1 adapters

| Source | Priority | Adapter |
|---|---|---|
| `cli` | 100 (human) | `openral prompt "do X"` — see [`openral_cli.prompt`](../../python/cli/src/openral_cli/prompt.py) |
| `dashboard` | 100 (human) | Future WebSocket adapter (post-v1) |
| `auto` | 10 (machine) | `EmitPromptTool` self-cascades from the F4 reasoner |

Per-source allowlist comes from the constructor's `sources` dict; the
deployment YAML may restrict the set (ADR-0018 §3.F10 "per-source
allowlist in deployment YAML").

## QoS

`/openral/prompt` and every `/openral/prompt_in/<source>` use
`RELIABLE + VOLATILE + KEEP_LAST=10` per ADR-0018 §1. No silent drops
— a saturated subscriber surfaces as a structlog warning, not a
swallowed message.

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

- [ADR-0018](../../docs/adr/0018-ros2-reasoner-supervisor.md) §3.F10.
- [`packages/openral_reasoner_ros`](../openral_reasoner_ros/) — F4 consumer.
- [`openral_cli.prompt`](../../python/cli/src/openral_cli/prompt.py) — `openral prompt` CLI entry point.
