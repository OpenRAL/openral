#!/usr/bin/env python3
"""Completion check node: hold the policy, ask a VLM whether the task is done, end or resume.

A VLA emits no success signal, and a reward model's success head is no completion signal on
its own (Spark R221-R223, 2026-10-22: Robometer's success peaked mid-carry as often as at the
placement). So the head is a cheap TRIGGER and a VLM the judge:

1. Trigger (``openral_reasoner.CompletionTrigger``): the reward monitor's success stream
   (``CriticScore`` whose ``critic_id`` ends in ``#success``) at or above
   ``success_threshold``, or a release -- a ``DETACH`` on ``/openral/grasp_events`` or a track
   newly ``released`` on ``/openral/tracked_objects`` -- at most once per ``cooldown_s``, only
   while ``/openral/reward/active_task`` names a running task, and never while
   ``/openral/attachment_state`` holds a payload (a release fires once the hand is empty).
   The question adds that a task putting an object somewhere is finished only once the robot
   let go of it (Spark R235 / R237: "yes" with the banana still in the gripper).
2. Hold: ``/openral/rskill/hold`` (``std_srvs/SetBool``). The policy keeps stepping; its
   commands are withheld, so the robot keeps its last command while the VLM looks.
3. Ask, by ``judge``:

   * ``scene_vlm``: ``/openral/perception/query_scene`` once per camera in ``cameras``, the
     reasoner's ``COMPLETION_QUESTION`` about the active task, ``parse_yes_no`` on each
     answer. Done only when every camera says yes (a miss resumes; a false yes would end).
   * ``reasoner``: the reasoner's own model (``build_tool_use_client_from_env``: the
     ``OPENRAL_REASONER_*`` env), one ``describe_image`` call on the cameras' latest frames
     tiled left to right in ``cameras`` order, ``parse_yes_no`` on the answer. Spark R233
     (2026-10-22): the 4B scene VLM thought ~30 s a question and often ran out of tokens.
4. Done: ``/openral/rskill/complete`` (``std_srvs/Trigger``) ends the goal as a success.
   Otherwise -- a no, a failed or timed-out query -- the hold is released.

Advisory (CLAUDE.md §1.1): it only withholds or ends motion, never commands any; every
command still crosses the safety kernel. A crash mid-check leaves the goal held, never moving,
until its deadline.

Parameters:
    success_threshold (float): trigger bar on the success head. Default 0.6.
    cooldown_s (float): seconds (node clock) between checks. Default 3.0: in a sim running at
        0.36x real time, 10 s of sim time outlasted a 9 s window the banana lay in the bowl
        (Spark R270, 2026-10-23), and the next check judged the bowl the arm had tipped since.
    cameras (str[]): ``query_scene`` cameras, all of which must say yes. Default
        ``["exterior", "wrist"]``.
    release_rechecks (int): times a release is asked again, one cooldown apart, while the
        hand stays empty (a 1 Hz tick offers them). Default 3: asked as the gripper opened,
        the judge said "no" to a banana lying in the bowl (Spark R329, 2026-10-23).
    settle_s (float): wall seconds between the hold and the first query (fresh frames).
        Default 0.5.
    query_timeout_s (float): wall seconds per service call. Default 90.0 (the scene VLM
        thinks before it answers: ~29 s a question on a GB10).
    judge (str): ``scene_vlm`` (default) or ``reasoner``.
    flip_180 (bool): rotate the ``reasoner`` judge's frames 180 deg (LIBERO/MuJoCo publish
        bottom-up; the launch mirrors ``OPENRAL_DASHBOARD_FLIP_180``). Default False.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import rclpy
from openral_observability.rclpy_spin import spin_executor_until_shutdown
from openral_reasoner.completion import (
    COMPLETION_QUESTION,
    CompletionTrigger,
    parse_yes_no,
    tiled_jpeg,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

_RELIABLE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class CompletionCheckNode(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
    """Trigger, hold, ask, then complete or resume (module docstring)."""

    def __init__(self) -> None:  # noqa: PLR0915  # reason: flat param/sub/client setup
        """Read params, subscribe the triggers, create the service clients and the worker."""
        super().__init__("openral_completion_check")
        self.declare_parameter("success_threshold", 0.6)
        self.declare_parameter("cooldown_s", 3.0)
        self.declare_parameter("release_rechecks", 3)
        self.declare_parameter("cameras", ["exterior", "wrist"])
        self.declare_parameter("settle_s", 0.5)
        self.declare_parameter("query_timeout_s", 90.0)
        self.declare_parameter("judge", "scene_vlm")
        self.declare_parameter("flip_180", False)
        gp = self.get_parameter
        self._trigger = CompletionTrigger(
            success_threshold=gp("success_threshold").get_parameter_value().double_value,
            cooldown_s=gp("cooldown_s").get_parameter_value().double_value,
            release_rechecks=gp("release_rechecks").get_parameter_value().integer_value,
        )
        self._cameras = [c for c in gp("cameras").get_parameter_value().string_array_value if c]
        self._settle_s = gp("settle_s").get_parameter_value().double_value
        self._timeout_s = gp("query_timeout_s").get_parameter_value().double_value
        self._judge = gp("judge").get_parameter_value().string_value
        self._flip_180 = gp("flip_180").get_parameter_value().bool_value
        if self._judge not in ("scene_vlm", "reasoner"):
            from openral_core.exceptions import ROSConfigError

            raise ROSConfigError(f"judge must be scene_vlm or reasoner, got {self._judge!r}")
        self._llm: Any = None
        self._frames: dict[str, Any] = {}
        if self._judge == "reasoner":
            from openral_core import camera_topic
            from openral_reasoner.tool_use import build_tool_use_client_from_env
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image

            self._llm = build_tool_use_client_from_env()  # fails at start, not mid-goal
            for cam in self._cameras:
                self.create_subscription(
                    Image, camera_topic(cam),
                    lambda m, c=cam: self._frames.__setitem__(c, m), qos_profile_sensor_data,
                )  # fmt: skip

        from openral_msgs.msg import AttachmentState, CriticScore, GraspEvent, TrackedObjectSet
        from openral_msgs.srv import QueryScene
        from std_msgs.msg import Bool, String
        from std_srvs.srv import SetBool, Trigger

        self._detach = int(GraspEvent.KIND_DETACH)
        self._query_cls, self._hold_cls, self._done_cls = QueryScene, SetBool, Trigger
        group = ReentrantCallbackGroup()
        self._hold = self.create_client(SetBool, "/openral/rskill/hold", callback_group=group)
        self._done = self.create_client(Trigger, "/openral/rskill/complete", callback_group=group)
        self._query = self.create_client(
            QueryScene, "/openral/perception/query_scene", callback_group=group
        )
        self._task = ""
        self._holding = False  # a hand holds a payload (/openral/attachment_state)
        # The object tracker's /openral/object_tracker/hands_clear; None until it publishes
        # (no tracker: the hand-clear gate is off).
        self._hands_clear: bool | None = None
        self._released: set[tuple[str, int]] = set()
        self._jobs: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self._busy = threading.Event()  # a check pending or in flight
        self.create_subscription(String, "/openral/reward/active_task", self._on_task, _RELIABLE)
        self.create_subscription(CriticScore, "/openral/critic/score", self._on_score, _RELIABLE)
        self.create_subscription(GraspEvent, "/openral/grasp_events", self._on_grasp, _RELIABLE)
        self.create_subscription(
            TrackedObjectSet, "/openral/tracked_objects", self._on_tracked, _RELIABLE
        )
        latched = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            AttachmentState, "/openral/attachment_state", self._on_attachments, latched
        )
        self.create_subscription(
            Bool, "/openral/object_tracker/hands_clear", self._on_hands_clear, _RELIABLE
        )
        self.create_timer(1.0, self._offer)  # the release rechecks have no event of their own
        threading.Thread(target=self._worker, name="completion_check", daemon=True).start()
        self.get_logger().info(
            f"completion_check: success>={self._trigger.success_threshold} or release, "
            f"cooldown {self._trigger.cooldown_s}s, {self._trigger.release_rechecks} release "
            f"rechecks, cameras={self._cameras}, judge={self._judge}"
        )

    # -- triggers ------------------------------------------------------------------------

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_task(self, msg: Any) -> None:
        self._task = (msg.data or "").strip()

    def _offer(self, *, success: float | None = None, released: bool = False) -> None:
        if self._busy.is_set():
            if released:
                self._trigger.mark_released()  # a check is running: this one fires next
            return
        if not self._task:
            return  # no running task
        # A hand still at the target counts as holding it: asked with the open fingers on
        # the banana in the bowl, the judge said done (Spark R336 / R338 / R340, 2026-10-23).
        holding = self._holding or self._hands_clear is False
        reason = self._trigger.fire(
            self._now_s(), success=success, released=released, holding=holding
        )
        if reason is not None:
            self._busy.set()
            self._jobs.put_nowait((reason, self._task))

    def _on_score(self, msg: Any) -> None:
        if str(msg.critic_id).endswith("#success"):
            self._offer(success=float(msg.score))

    def _on_grasp(self, msg: Any) -> None:
        if int(msg.kind) == self._detach:
            self._offer(released=True)

    def _on_attachments(self, msg: Any) -> None:
        self._holding = bool(msg.objects)
        if not self._holding:
            self._offer()  # a release seen while the payload was still on the wire

    def _on_hands_clear(self, msg: Any) -> None:
        self._hands_clear = bool(msg.data)

    def _on_tracked(self, msg: Any) -> None:
        now = {(o.object_id, int(o.epoch)) for o in msg.objects if o.status == "released"}
        fresh = now - self._released
        self._released |= now
        if fresh:
            self._offer(released=True)

    # -- the check -----------------------------------------------------------------------

    def _call(self, client: Any, request: Any) -> Any:  # reason: rclpy service types
        """One service call from the worker thread; ``None`` on no service or a timeout."""
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        return future.result() if done.wait(self._timeout_s) else None

    def _set_hold(self, on: bool) -> bool:
        req = self._hold_cls.Request()
        req.data = on
        rep = self._call(self._hold, req)
        return bool(rep is not None and rep.success)

    def _worker(self) -> None:
        while rclpy.ok():
            reason, task = self._jobs.get()
            try:
                self._check(reason, task)
            except Exception as exc:  # reason: one failed check must not end the worker
                # A failed service future re-raises in _call; uncaught, it ended the thread
                # and every later trigger was dropped silently (#421 review).
                self.get_logger().error(f"completion_check {reason}: check failed: {exc!r}")
            finally:
                self._busy.clear()

    def _check(self, reason: str, task: str) -> None:
        if not self._set_hold(True):
            self.get_logger().warning(f"completion_check {reason}: hold refused; not checking")
            return
        threading.Event().wait(self._settle_s)
        answers: list[str] = []
        release = True
        try:
            labels = self._ask(task, answers)
            done = bool(answers) and all(parse_yes_no(a) for a in answers)
            self.get_logger().info(
                f"completion_check {reason}: done={done} judge={self._judge} task={task!r} "
                + " ".join(f"{c}={a[:80]!r}" for c, a in zip(labels, answers, strict=True))
            )
            if done:
                rep = self._call(self._done, self._done_cls.Request())
                release = rep is None or not rep.success  # completed: its reset clears the hold
                if release:
                    self.get_logger().warning("completion_check: complete refused; resuming")
        finally:
            if release:
                self._set_hold(False)

    def _ask(self, task: str, answers: list[str]) -> list[str]:
        """Ask the judge; append its answers (``""`` = no answer); return their labels."""
        if self._judge == "scene_vlm":
            for camera in self._cameras:
                req = self._query_cls.Request()
                req.question = _question(task)
                req.camera = camera
                rep = self._call(self._query, req)
                answers.append(rep.answer if rep is not None and rep.ok else "")
            return list(self._cameras)
        if any(c not in self._frames for c in self._cameras):
            answers.append("")  # a camera never published: no verdict
            return ["tiled"]
        frames = [self._frames[c] for c in self._cameras]
        question = (
            f"The image shows {len(frames)} camera views of the robot side by side, left to "
            f"right: {', '.join(self._cameras)}. " + _question(task)
        )
        try:
            jpeg = tiled_jpeg(
                [(bytes(f.data), f.height, f.width, f.encoding) for f in frames],
                flip_180=self._flip_180,
            )
            answers.append(self._llm.describe_image(image_jpeg=jpeg, question=question))
        except Exception as exc:  # reason: provider / transport failures resume the task
            self.get_logger().warning(f"completion_check: reasoner judge failed: {exc}")
            answers.append("")
        return ["tiled"]


def _question(task: str) -> str:
    """The reasoner's ``COMPLETION_QUESTION``, with the release rule spelled out.

    Spark R235 / R237 (2026-10-22): asked only whether the task was finished, the judge said
    yes with the banana still in the gripper over the bowl.
    """
    return COMPLETION_QUESTION.format(task=task) + (
        " A task that puts an object somewhere is finished only once the robot has let go of"
        " it: if the gripper still holds the object, answer 'no'."
    )


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the completion check node, shut down cleanly."""
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = CompletionCheckNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        try:
            spin_executor_until_shutdown(executor)
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
