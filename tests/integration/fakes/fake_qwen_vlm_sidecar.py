"""A Qwen scene-VLM sidecar stand-in at its process + network boundary (ZMQ REP + msgpack).

Started by ``QwenSceneVlm``'s auto-spawn in place of ``tools/qwen_vlm_sidecar.py``
(``OPENRAL_QWEN_VLM_SIDECAR``), with that script's argv. Answers ``ping`` and writes its
pid to ``FAKE_QWEN_SIDECAR_PID_FILE`` so a test can see whether it outlived its client.
Never exits on its own: only a terminate/kill from the client ends it. Under
``tests/integration/fakes/`` per CLAUDE.md §1.11; production code never imports it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import msgpack
import zmq


def main() -> None:
    """Serve ``ping`` on ``--host``/``--port`` until killed."""
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    args, _ = p.parse_known_args()
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    Path(os.environ["FAKE_QWEN_SIDECAR_PID_FILE"]).write_text(str(os.getpid()))
    while True:
        msgpack.unpackb(sock.recv(), raw=False)
        sock.send(msgpack.packb({"ok": True}, use_bin_type=True))


if __name__ == "__main__":
    main()
