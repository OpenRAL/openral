"""read_frame_trace — pivot a LeRobotDataset frame back to its OTel ids.

ISSUE-109: every frame written by :class:`LeRobotDatasetSink` carries the
``trace_id`` / ``span_id`` of the ``rskill.tick`` span that produced it.
This module reads those columns back so ``openral replay --frame
<repo>/<ep>/<frame>`` can resolve a dataset coordinate to a trace and
join it against the bag + dashboard spans.

The read goes straight at the v3 parquet shards (``pyarrow``) rather than
through :class:`lerobot.datasets.LeRobotDataset` — indexing the dataset
would decode the episode MP4s, which needs a working torchcodec/ffmpeg
backend. The correlation columns are plain ``string`` parquet values, so
a dependency-light read keeps the pivot usable on any host.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from openral_core.exceptions import ROSConfigError

__all__ = ["read_frame_trace"]

_log = structlog.get_logger(__name__)


def read_frame_trace(*, root: Path | str, episode_idx: int, frame_idx: int) -> tuple[str, str]:
    """Return the ``(trace_id, span_id)`` of one LeRobotDataset frame.

    Args:
        root: Dataset root directory (the ``--root`` a v3 dataset was
            written to). The function reads ``root/data/**/*.parquet``.
        episode_idx: Zero-based ``episode_index`` of the target frame.
        frame_idx: Zero-based ``frame_index`` within that episode.

    Returns:
        ``(trace_id, span_id)`` — the 32-hex / 16-hex ids stamped on the
        frame at write time. Either may be ``""`` if the producing tick
        had no valid OTel span.

    Raises:
        ROSConfigError: When ``root`` holds no v3 parquet data, the
            dataset predates the ISSUE-109 correlation columns, or no row
            matches ``(episode_idx, frame_idx)``.

    Example:
        >>> from openral_dataset import read_frame_trace
        >>> read_frame_trace(root="/tmp/ds", episode_idx=0, frame_idx=0)  # doctest: +SKIP
        ('a8f1049a749b...', 'b3c2...')
    """
    import pyarrow.parquet as pq

    root = Path(root)
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise ROSConfigError(
            f"read_frame_trace: no LeRobot dataset parquet under {root}/data; "
            "is the dataset root correct?"
        )

    for parquet_path in files:
        # reason: pyarrow.parquet.read_table is untyped upstream despite py.typed
        table = pq.read_table(parquet_path)  # type: ignore[no-untyped-call]
        columns = set(table.column_names)
        if "trace_id" not in columns or "span_id" not in columns:
            raise ROSConfigError(
                f"read_frame_trace: dataset at {root} has no trace_id/span_id columns; "
                "it predates the ISSUE-109 per-frame correlation fields and cannot be "
                "pivoted. Re-record with a current LeRobotDatasetSink."
            )
        rows = table.to_pylist()
        for row in rows:
            if int(row["episode_index"]) == episode_idx and int(row["frame_index"]) == frame_idx:
                return str(row["trace_id"]), str(row["span_id"])

    raise ROSConfigError(
        f"read_frame_trace: no frame at episode_idx={episode_idx} frame_idx={frame_idx} "
        f"in the dataset under {root}"
    )
