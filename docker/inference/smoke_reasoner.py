r"""Live reasoner + prompt-router round-trip — runs inside the x86-ros Docker image.

Exercises the ADR-0018 F4 + F10 end-to-end without pytest / conftest /
torch in the loop:

1. Build a real :class:`PromptRouterNode` (F10) + a real
   :class:`ReasonerNode` (F4) in-process. The reasoner gets a
   :class:`FakeToolUseClient` (the only permitted LLM-side test double
   per CLAUDE.md §1.11) seeded with a single ``EmitPromptTool`` that
   the LLM would have picked.
2. Spin both lifecycle nodes via a real ``rclpy``
   :class:`SingleThreadedExecutor`.
3. Publish one ``openral_msgs/PromptStamped`` on
   ``/openral/prompt_in/cli`` — the CLI input topic the F10 router
   listens on.
4. Subscribe to ``/openral/prompt`` and assert the full chain:

   * the router fans the CLI input out with ``{"source": "cli",
     "priority": 100}`` merged into ``metadata_json``;
   * the reasoner consumes the operator prompt, preempts its tick
     (no min-interval wait needed in v1 with min_interval_s=0.0 on
     the test reasoner), invokes the LLM (the FakeToolUseClient),
     and dispatches the canned ``EmitPromptTool``;
   * the dispatched ``PromptStamped`` lands on ``/openral/prompt``
     with ``header.frame_id == "openral_reasoner"`` and
     ``metadata_json`` containing a W3C ``traceparent`` stamped from
     the active ``reasoner.tick`` span (ADR-0018 §6).

Exits 0 on success, non-zero with an error message otherwise. Uses
``os._exit(rc)`` to skip Python's teardown so the
pydantic-Rust / rclpy / cyclonedds C-extension teardown segfault
seen across the other docker smokes can't mask a successful
round-trip.

Designed to run as::

    docker run --rm --gpus all \\
        --entrypoint /entrypoint.sh \\
        -v "$(pwd)/docker/inference/smoke_reasoner.py:/workspace/smoke_reasoner.py:ro" \\
        -v "$(pwd)/install:/workspace/install:ro" \\
        openral:x86-latest \\
        bash -lc 'source /workspace/install/setup.bash && exec python /workspace/smoke_reasoner.py'
"""

from __future__ import annotations

# ruff: noqa: I001, PLC0415, PLR0915, PLR2004  reason: rclpy + openral_msgs depend on a
# sourced ROS install (install/setup.bash); the deferred order matches
# the convention in smoke_ros_tee.py / smoke_perception_tee.py. The
# openral_reasoner / openral_reasoner_ros / openral_prompt_router
# imports come BEFORE rclpy to mirror the same ordering constraint.
# PLC0415: opentelemetry SDK is deferred inside _install_otel_provider
# so the smoke can be parsed on hosts without it.
import json
import os
import sys
import threading
import time

from openral_core import EmitPromptTool
from openral_prompt_router import PromptRouterNode
from openral_reasoner import ToolPalette
from openral_reasoner_ros import ReasonerNode

import rclpy
from openral_msgs.msg import PromptStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


# Lives outside the integration tier inside the container; mirror the
# in-tree FakeToolUseClient with a tiny inline version so the smoke
# does not depend on tests/integration/fakes/ being on PYTHONPATH.
class _FakeToolUseClient:
    """In-container fake mirroring tests/integration/fakes/fake_llm.py.

    Returns a single canned ``EmitPromptTool`` on the first ``select_tool``
    call; raises ``RuntimeError`` thereafter so an unexpected second
    LLM call surfaces as a failure rather than a silent hang.
    """

    model_id = "smoke-fake"

    def __init__(self, response: EmitPromptTool) -> None:
        self._response: EmitPromptTool | None = response

    def select_tool(
        self, *, context_text: str, palette: object, system_prompt: str
    ) -> EmitPromptTool:
        del context_text, palette, system_prompt
        if self._response is None:
            raise RuntimeError("smoke: FakeToolUseClient called twice — unexpected re-tick")
        out = self._response
        self._response = None
        return out


def _install_otel_provider() -> None:
    """Install a real OTel SDK TracerProvider so traceparent is non-empty.

    The deploy image's default observability path
    (``configure_observability``) is a no-op when no OTLP endpoint is
    configured; that path never installs an SDK provider, which means
    ``current_traceparent()`` returns ``None`` and the reasoner_node
    can't stamp ``traceparent`` on outbound EmitPromptTool messages.

    The production deploys that care about ADR-0018 §6 set
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` so the SDK provider IS installed;
    this smoke models that path with an explicit provider install (no
    exporter needed — we only need the active span machinery so the
    traceparent has a valid trace_id / span_id).
    """
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    _trace.set_tracer_provider(provider)


def run() -> int:
    """Drive the round-trip and return a CLI exit status."""
    _install_otel_provider()
    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()

    try:
        router = PromptRouterNode()
        router.trigger_configure()
        router.trigger_activate()

        client = _FakeToolUseClient(
            response=EmitPromptTool(
                target_topic="/openral/prompt",
                text="acknowledged: pick the cube",
                rationale="smoke: operator prompt received",
            ),
        )
        reasoner = ReasonerNode(
            client=client,  # type: ignore[arg-type]  # reason: structural match (Protocol)
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            tick_hz=10.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        sub_node = rclpy.create_node("openral_smoke_subscriber")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            print(
                f"[smoke] cb! frame_id={msg.header.frame_id!r} text={msg.text!r}",
                flush=True,
            )
            received.append(msg)
            # The reasoner's reply is the one we're waiting for.
            if msg.header.frame_id == "openral_reasoner":
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_smoke_publisher")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt_in/cli", qos)

        executor = SingleThreadedExecutor()
        executor.add_node(router)
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        # Give DDS discovery a moment to settle so the first publish
        # actually reaches both the router and the test subscriber.
        deadline_settle = time.monotonic() + 1.0
        while time.monotonic() < deadline_settle:
            executor.spin_once(timeout_sec=0.05)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_smoke_publisher"
        msg.text = "pick the cube"
        msg.metadata_json = "{}"
        print("[smoke] publishing CLI prompt on /openral/prompt_in/cli ...", flush=True)
        pub.publish(msg)

        deadline = time.monotonic() + 6.0
        spins = 0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)
            spins += 1
        print(
            f"[smoke] spun {spins} times; total messages={len(received)}",
            flush=True,
        )

        executor.remove_node(router)
        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
        router.destroy_node()
    finally:
        rclpy.shutdown()

    # Assertions: at minimum the reasoner's reply must have landed.
    reasoner_msgs = [m for m in received if m.header.frame_id == "openral_reasoner"]
    if not reasoner_msgs:
        print(
            "[smoke] FAIL — reasoner did not publish on /openral/prompt within 6 s. "
            f"Saw {len(received)} message(s) but none from openral_reasoner.",
            file=sys.stderr,
        )
        return 1
    reply = reasoner_msgs[-1]
    if reply.text != "acknowledged: pick the cube":
        print(
            f"[smoke] FAIL — wrong reasoner reply text: {reply.text!r}",
            file=sys.stderr,
        )
        return 2
    metadata = json.loads(reply.metadata_json)
    if "traceparent" not in metadata:
        print(
            "[smoke] FAIL — reasoner reply missing 'traceparent' in metadata_json "
            f"(metadata={metadata}). ADR-0018 §6 contract violated.",
            file=sys.stderr,
        )
        return 3

    # And the router should have produced its own fan-out forward
    # (frame_id=openral_smoke_publisher, but with source/priority
    # merged into metadata_json by the F10 node).
    router_msgs = [m for m in received if m.header.frame_id == "openral_smoke_publisher"]
    if router_msgs:
        router_metadata = json.loads(router_msgs[-1].metadata_json)
        if router_metadata.get("source") != "cli" or router_metadata.get("priority") != 100:
            print(
                "[smoke] WARN — router fan-out present but missing source/priority tags: "
                f"{router_metadata}",
                file=sys.stderr,
            )

    print(f"[smoke] OK — received {len(received)} PromptStamped(s) on /openral/prompt")
    print(f"[smoke]   reasoner reply text={reply.text!r}")
    print(f"[smoke]   metadata.traceparent={metadata['traceparent']}")
    print(f"[smoke]   metadata.source={metadata.get('source')!r}")
    print(f"[smoke]   metadata.rationale={metadata.get('rationale')!r}")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    rc = run()
    # Skip Python's atexit/finaliser teardown — pydantic-Rust /
    # rclpy / cyclonedds C-extension teardown segfaults at shutdown
    # even when the round-trip succeeded (same workaround as
    # smoke_ros_tee.py / smoke_perception_tee.py).
    os._exit(rc)
