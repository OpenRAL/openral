"""Compliance recorder: the ROS graph's failures and safety stops as compliance records.

Issue #467. The recorder node subscribes to the failure bus and the latched
``/openral/safety_status`` topic and emits ``openral_core.ComplianceRecord``s
through ``openral_observability.compliance.ComplianceEmitter`` (OTLP log records).
"""
