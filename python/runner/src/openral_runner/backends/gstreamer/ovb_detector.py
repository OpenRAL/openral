"""On-demand locator backed by the Open Vision Banana (OVB) sidecar.

Open Vision Banana (arXiv 2611.15844) fine-tunes FLUX.2 [klein] into one
image-to-image model that *paints* its perception answers: for referring
segmentation it paints the region matching a description in one colour and
everything else black. It needs a pinned ``diffusers`` / ``transformers==4.57``
stack the runtime's ``transformers>=5`` cannot share, so it runs in an isolated
sidecar (``tools.ovb_sidecar``); this backend is the ZMQ client.

The split mirrors ``LocateAnythingDetector``: the sidecar returns the raw painted
image, and everything that decides what a detection *is* — the prompt, the
colour decode, the connected-region split, the box filter — lives here, where
it is unit-testable without a GPU or the sidecar venv.

It implements ``detect(frame_bgr, width, height, sensor_id)`` and
``detect_with_query(...)`` like the other detector backends, so the
``locate_in_view`` service drives it unchanged.

**Confidence.** OVB paints a mask; it emits no score. Every detection is
reported at ``confidence=1.0`` rather than a fabricated number (CLAUDE.md §1.2).

**Depth is out of scope.** OVB's depth output is relative inverse depth, never
metric; this backend only runs the referring-segmentation task.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from openral_core import ObjectDetection2D, ObjectsMetadata
from openral_core.exceptions import ROSConfigError

from ._zmq_sidecar import ZmqSidecarMixin

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "REFSEG_PAINT_RGB",
    "OpenVisionBananaDetector",
    "build_objects_metadata_from_paint",
    "build_refseg_prompt",
    "decode_painted_mask",
    "mask_regions_xyxy",
]

# The paint colour of the upstream quick-start referring-segmentation prompt
# (OpenVisionBanana PROMPTS.md). The model was trained on many colours; this
# one is kept so the prompt matches the documented example exactly.
REFSEG_PAINT_RGB: tuple[int, int, int] = (220, 30, 30)

# A region smaller than this fraction of the frame is diffusion speckle, not an
# object; the world-state lift would project it to a spurious point.
_MIN_REGION_FRAC = 0.001


def build_refseg_prompt(expression: str, paint_rgb: tuple[int, int, int] = REFSEG_PAINT_RGB) -> str:
    """Build OVB's referring-segmentation prompt for ``expression``.

    Args:
        expression: Free-text referring expression ("the red mug on the left").
        paint_rgb: Colour the model is asked to paint the matching region.

    Returns:
        The task-tagged prompt, in the upstream quick-start wording.

    Raises:
        ROSConfigError: If ``expression`` is empty or contains a double quote,
            which would break the quoted description in the prompt.

    Example:
        >>> build_refseg_prompt("cup").startswith("[TASK: REFERRING_SEGMENTATION]")
        True
    """
    expr = expression.strip()
    if not expr:
        raise ROSConfigError("OVB referring expression must be non-empty")
    if '"' in expr:
        raise ROSConfigError(f"OVB referring expression must not contain '\"': {expr!r}")
    r, g, b = paint_rgb
    return (
        "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this "
        f'description: "{expr}". Paint the matching region red [{r}, {g}, {b}] and '
        "leave everything else black [0, 0, 0]. Output only the segmentation mask image."
    )


def decode_painted_mask(
    rgb: npt.NDArray[np.uint8], paint_rgb: tuple[int, int, int] = REFSEG_PAINT_RGB
) -> npt.NDArray[np.bool_]:
    """Decode a painted ``H x W x 3`` RGB answer into a boolean foreground mask.

    Each pixel snaps to the nearer of the paint colour or black in RGB — the
    rule upstream's referring-segmentation evaluation uses
    (``rgb_to_foreground``), so anti-aliased edges split at the midpoint.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((2, 2, 3), np.uint8)
        ... img[0, 0] = (200, 40, 40)
        >>> decode_painted_mask(img).tolist()
        [[True, False], [False, False]]
    """
    import numpy as np  # noqa: PLC0415 — lazy: keep numpy off the import path

    arr = rgb.astype(np.int64)
    d_fg = ((arr - np.asarray(paint_rgb, dtype=np.int64)) ** 2).sum(-1)
    d_bg = (arr**2).sum(-1)
    mask: npt.NDArray[np.bool_] = d_fg < d_bg
    return mask


def mask_regions_xyxy(
    mask: npt.NDArray[np.bool_], *, min_region_frac: float = _MIN_REGION_FRAC
) -> list[tuple[int, int, int, int]]:
    """Split a mask into 4-connected regions and return each one's pixel box.

    Boxes are ``(x1, y1, x2, y2)`` with exclusive ``x2``/``y2``, ordered by first
    appearance in a row-major scan. Regions under ``min_region_frac`` of the
    frame are dropped as speckle.

    Example:
        >>> import numpy as np
        >>> m = np.zeros((4, 6), bool)
        ... m[0:2, 0:2] = True
        ... m[2:4, 4:6] = True
        >>> mask_regions_xyxy(m, min_region_frac=0.0)
        [(0, 0, 2, 2), (4, 2, 6, 4)]
    """
    import numpy as np  # noqa: PLC0415 — lazy: keep numpy off the import path

    h, w = mask.shape
    min_px = max(1, round(min_region_frac * h * w))
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    # ponytail: Python BFS over foreground pixels only — milliseconds at the model's
    # output size (callers label before upscaling), ~2.4 s on a large 1080p mask.
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue = deque([(int(y0), int(x0))])
        count, x1, y1, x2, y2 = 0, int(x0), int(y0), int(x0), int(y0)
        while queue:
            y, x = queue.popleft()
            count += 1
            x1, x2, y1, y2 = min(x1, x), max(x2, x), min(y1, y), max(y2, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if count >= min_px:
            boxes.append((x1, y1, x2 + 1, y2 + 1))
    return boxes


def build_objects_metadata_from_paint(
    rgb: npt.NDArray[np.uint8],
    *,
    label: str,
    model_id: str,
    sensor_id: str,
    frame_size: tuple[int, int] | None = None,
) -> ObjectsMetadata | None:
    """Turn a painted answer into ``ObjectsMetadata`` in the frame's pixel space.

    Regions are found at the painted image's own resolution (the model's output,
    e.g. 224 px) and their boxes scaled to ``frame_size`` (``(width, height)``,
    default the painted size) — upscaling the mask first would multiply the
    region search by the area ratio for no extra information.

    Returns ``None`` when nothing was painted — an empty answer, not an error.
    """
    ph, pw = rgb.shape[:2]
    w, h = frame_size if frame_size is not None else (pw, ph)
    sx, sy = w / pw, h / ph
    dets = [
        ObjectDetection2D(
            label=label,
            confidence=1.0,
            bbox_xyxy=(
                round(x1 * sx),
                round(y1 * sy),
                min(w, round(x2 * sx)),
                min(h, round(y2 * sy)),
            ),
        )
        for x1, y1, x2, y2 in mask_regions_xyxy(decode_painted_mask(rgb))
    ]
    if not dets:
        return None
    return ObjectsMetadata(
        sensor_id=sensor_id, detections=dets, model_id=model_id, frame_width=w, frame_height=h
    )


def _find_sidecar_script() -> Path:
    """Locate ``tools/ovb_sidecar.py`` (env override or repo walk)."""
    override = os.environ.get("OPENRAL_OVB_SIDECAR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "ovb_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        "could not locate tools/ovb_sidecar.py; set OPENRAL_OVB_SIDECAR to its path"
    )


class OpenVisionBananaDetector(ZmqSidecarMixin):
    """ZMQ client + auto-managed lifecycle for the OVB sidecar.

    Same lifecycle as ``LocateAnythingDetector``: connect lazily on the first
    query, spawn the sidecar if nothing answers the ping, and tear down only a
    child this instance started.
    """

    def __init__(
        self,
        *,
        labels: list[str],
        model_id: str,
        weights_source: str,
        base_pipeline: str,
        transformer_subfolder: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = 0,
        max_side: int = 1024,
        host: str = "127.0.0.1",
        port: int = 5759,
        auto_spawn: bool = True,
        boot_timeout_s: float = 1800.0,
        request_timeout_s: float = 300.0,
    ) -> None:
        """Store config; the sidecar connection is deferred to the first query."""
        if not labels:
            raise ROSConfigError("OpenVisionBananaDetector requires at least one label")
        self.kind = "objects"
        self._query = labels[0]
        self._model_id = model_id
        self._weights_source = weights_source
        self._base_pipeline = base_pipeline
        self._transformer_subfolder = transformer_subfolder
        self._steps = num_inference_steps
        self._guidance = guidance_scale
        self._seed = seed
        self._max_side = max_side
        self._host = host
        self._port = port
        self._auto_spawn = auto_spawn
        self._boot_timeout_s = boot_timeout_s
        self._request_timeout_ms = int(request_timeout_s * 1000)
        self._zmq = None
        self._ctx = None
        self._sock = None
        self._child: subprocess.Popen[bytes] | None = None

    def _ensure_ready(self) -> None:
        """Connect to the sidecar (spawning it if needed) on first use."""
        if self._sock is not None:
            return
        try:
            import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy: keep zmq off the module import path
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            raise ROSConfigError(
                "Open Vision Banana needs the ZMQ + msgpack sidecar client; "
                "install it with `just sync --group ovb`."
            ) from exc
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._connect()
        if not self._try_ping():
            if not self._auto_spawn:
                raise ROSConfigError(
                    f"no OVB sidecar at tcp://{self._host}:{self._port} and auto_spawn=False"
                )
            self._spawn_and_wait()

    def _spawn_and_wait(self) -> None:
        import sys  # noqa: PLC0415 — lazy: only needed on the auto-spawn path

        cmd = [
            sys.executable,
            str(_find_sidecar_script()),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--model",
            self._weights_source,
            "--subfolder",
            self._transformer_subfolder,
            "--base",
            self._base_pipeline,
        ]
        print(f"[ovb-detector] spawning sidecar: {' '.join(cmd)}", flush=True)
        self._child = subprocess.Popen(cmd)
        deadline = time.monotonic() + self._boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(f"OVB sidecar exited early (code {self._child.returncode})")
            if self._try_ping():
                print("[ovb-detector] sidecar ready", flush=True)
                return
            time.sleep(2.0)
        raise ROSConfigError(f"OVB sidecar not ready within {self._boot_timeout_s}s")

    def set_query(self, text: str) -> None:
        """Override the persistent referring expression."""
        if not text.strip():
            raise ROSConfigError("detection query must be non-empty")
        self._query = text.strip()

    def detect(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str
    ) -> ObjectsMetadata | None:
        """Locate the persistent query in a raw BGR frame."""
        return self.detect_with_query(frame_bgr, width, height, sensor_id, self._query)

    def detect_with_query(
        self, frame_bgr: bytes, width: int, height: int, sensor_id: str, query: str
    ) -> ObjectsMetadata | None:
        """One-shot referring segmentation for ``query``; boxes one per painted region."""
        import numpy as np  # noqa: PLC0415 — lazy: keep numpy/PIL off the import path
        from PIL import Image  # noqa: PLC0415

        expression = query or self._query
        prompt = build_refseg_prompt(expression)  # validate before touching the sidecar
        self._ensure_ready()

        rgb = np.frombuffer(frame_bgr, dtype=np.uint8).reshape(height, width, 3)[:, :, ::-1]
        image = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
        longest = max(width, height)
        if longest > self._max_side:
            scale = self._max_side / longest
            image = image.resize(
                (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS
            )
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        reply = self._rpc(
            {
                "op": "generate",
                "image": buf.getvalue(),
                "prompt": prompt,
                "steps": self._steps,
                "guidance": self._guidance,
                "seed": self._seed,
            }
        )
        if not reply.get("ok"):
            raise ROSConfigError(f"OVB sidecar error: {reply.get('error')}")
        painted = Image.open(io.BytesIO(reply["image"])).convert("RGB")
        return build_objects_metadata_from_paint(
            np.asarray(painted, dtype=np.uint8),
            label=expression,
            model_id=self._model_id,
            sensor_id=sensor_id,
            frame_size=(width, height),
        )

    def close(self) -> None:
        """Close the socket and terminate the sidecar if we spawned it."""
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._child is not None and self._child.poll() is None:
            with contextlib.suppress(Exception):
                self._rpc({"op": "shutdown"}, recv_timeout_ms=2000)
            try:
                self._child.terminate()
                self._child.wait(timeout=10)
            except Exception:  # best-effort teardown; escalate to kill
                self._child.kill()
            self._child = None
