"""ADR-0018 §5 bullet 2 — human-driven estop forwarder.

The dashboard / Slack / voice adapters publish on
``/openral/human_estop``; the forwarder node republishes onto the
canonical ``/openral/estop`` topic and tags the event with a
``FailureTrigger(KIND_HUMAN, HumanEvidence(channel=...))`` so the
reasoner sees a structured event rather than just the bare estop.
"""
