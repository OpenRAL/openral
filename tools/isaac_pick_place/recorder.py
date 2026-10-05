"""Record a run: scene camera + head camera side by side, with the vision leg's attach
state, its declarations and the kernel status burnt in. Pipes rawvideo to ffmpeg.

Usage: recorder.py <out.mp4>  (stops on SIGINT / SIGTERM)
"""

import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import rclpy
from numpy.typing import NDArray
from openral_msgs.msg import AttachmentState, SafetyStatus
from PIL import Image, ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage

_H = 384
_FPS = 10
_W = 2 * 512


def _img(msg: RosImage) -> NDArray[np.uint8]:
    return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]


def main(out: str) -> None:
    rclpy.init()
    node = Node("pick_place_recorder")
    lock = threading.Lock()
    frames: dict[str, NDArray[np.uint8] | None] = {"top": None, "head": None}
    state = {
        "att": "attached: -",
        "grasp": "grasp region: -",
        "place": "place region: -",
        "kernel": "kernel: -",
    }

    def on_frame(key: str) -> Callable[[RosImage], None]:
        def cb(m: RosImage) -> None:
            with lock:
                frames[key] = _img(m)

        return cb

    def on_att(m: AttachmentState) -> None:
        objs = ", ".join(
            f"{o.object_id.split(':')[-1]}@{o.attach_link} [{o.evidence_kind}]" for o in m.objects
        )
        g, p = m.grasp_declaration, m.place_declaration
        grasp = "declared" if m.grasp_declaration_valid else "-"
        if m.grasp_declaration_valid and g.region_valid:
            grasp = "MEASURED " + ",".join(g.contact_links)
        place = "declared" if m.place_declaration_valid else "-"
        if m.place_declaration_valid and p.region_valid:
            place = "MEASURED"
        with lock:
            state["att"] = f"attached: {objs or 'none'}"
            state["grasp"] = f"grasp region: {grasp}"
            state["place"] = f"place region: {place}"

    def on_status(m: SafetyStatus) -> None:
        with lock:
            state["kernel"] = f"kernel: {'LATCHED ' if m.latched else ''}{m.detail[:70]}"

    latched = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
    )
    sensor = qos_profile_sensor_data
    node.create_subscription(RosImage, "/openral/cameras/top/image", on_frame("top"), sensor)
    node.create_subscription(RosImage, "/openral/cameras/head_zed/image", on_frame("head"), sensor)
    node.create_subscription(AttachmentState, "/openral/attachment_state", on_att, latched)
    node.create_subscription(SafetyStatus, "/openral/safety_status", on_status, latched)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())  # after rclpy.init, which hooks SIGINT
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{_W}x{_H + 90}", "-r", str(_FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "frag_keyframe+empty_moov", out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    assert ff.stdin is not None
    t0 = time.time()
    try:
        while not stop.is_set():
            time.sleep(1.0 / _FPS)
            with lock:
                shown = (frames["top"], frames["head"])
                lines = [state["att"], state["grasp"], state["place"], state["kernel"]]
            canvas = Image.new("RGB", (_W, _H + 90), (20, 20, 20))
            for i, frame in enumerate(shown):
                if frame is not None:
                    canvas.paste(Image.fromarray(frame).resize((512, _H)), (i * 512, 0))
            d = ImageDraw.Draw(canvas)
            d.text(
                (8, _H + 4),
                f"t={time.time() - t0:6.1f}s   left: scene camera   right: head_zed colour",
                fill=(200, 200, 200),
            )
            for j, line in enumerate(lines):
                d.text((8, _H + 22 + 16 * j), line, fill=(255, 220, 120))
            ff.stdin.write(canvas.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()


if __name__ == "__main__":
    main(sys.argv[1])
