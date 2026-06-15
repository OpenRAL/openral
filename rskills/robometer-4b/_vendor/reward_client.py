"""Phase 3 PoC client: stream a real rollout video to the reward sidecar and
query windowed progress/success as it unfolds. Demonstrates end-to-end that the
NF4 sidecar produces a rising progress signal on a successful rollout.

Run (sidecar must be up): /tmp/robometer-env/bin/python rskills/robometer-4b/_vendor/reward_client.py \
    --video /tmp/robometer_example.mp4 --task "Put green stick in brown bowl" --port 5599 --fps 3
"""

from __future__ import annotations

import argparse

import decord
import msgpack
import numpy as np
import zmq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--chunk", type=int, default=2, help="frames per ingest step")
    args = ap.parse_args()

    vr = decord.VideoReader(args.video)
    src_fps = vr.get_avg_fps()
    step = max(1, int(round(src_fps / args.fps)))
    idxs = list(range(0, len(vr), step))
    frames = vr.get_batch(idxs).asnumpy().astype(np.uint8)  # (N, H, W, 3)
    print(f"[client] {len(vr)} src frames @ {src_fps:.1f}fps -> {len(frames)} sampled @ ~{args.fps}fps")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")

    def call(d: dict) -> dict:
        sock.send(msgpack.packb(d))
        return msgpack.unpackb(sock.recv(), raw=False)

    print(f"[client] set_task -> {call({'cmd': 'set_task', 'task': args.task})}")

    dt_ns = int(1e9 / args.fps)
    print(f"\n{'frames':>7} | {'progress_now':>12} | {'success_now':>11} | {'prog_trend':>10} | {'stalled':>7}")
    print("-" * 60)
    fed = 0
    for start in range(0, len(frames), args.chunk):
        chunk = frames[start:start + args.chunk]
        stamp_ns = fed * dt_ns
        call({"cmd": "ingest", "frames": chunk.tobytes(),
              "shape": list(chunk.shape), "stamp_ns": stamp_ns, "fps": args.fps})
        fed += len(chunk)
        q = call({"cmd": "query", "window_s": 30.0})
        if q.get("stale") and not q.get("frames_seen"):
            continue
        print(f"{q['frames_seen']:>7} | {q['progress_now']:>12.4f} | {q['success_now']:>11.4f} "
              f"| {q['progress_trend']:>10.4f} | {str(q['stalled']):>7}")

    # final full-window series
    final = call({"cmd": "query", "window_s": 30.0})
    print(f"\n[client] final progress series: {final['progress_series']}")
    print(f"[client] final success  series: {final['success_series']}")
    call({"cmd": "shutdown"})
    sock.close()
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
