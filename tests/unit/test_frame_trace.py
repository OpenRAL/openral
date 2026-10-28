"""``read_frame_trace`` decodes only the pivot columns of the matching rows (CWE-409)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_dataset import read_frame_trace

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_shard(root: Path, *, with_trace: bool = True) -> None:
    shard = root / "data" / "chunk-000" / "file-000.parquet"
    shard.parent.mkdir(parents=True)
    columns = {
        "episode_index": pa.array([0, 0, 1, 1], pa.int64()),
        "frame_index": pa.array([0, 1, 0, 1], pa.int64()),
        "observation.state": pa.array([[0.0] * 7] * 4, pa.list_(pa.float32())),
    }
    if with_trace:
        columns["trace_id"] = pa.array([f"{i:032x}" for i in range(4)])
        columns["span_id"] = pa.array([f"{i:016x}" for i in range(4)])
    pq.write_table(pa.table(columns), shard)


def test_finds_the_matching_frame(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    assert read_frame_trace(root=tmp_path, episode_idx=1, frame_idx=0) == (
        f"{2:032x}",
        f"{2:016x}",
    )


def test_missing_frame_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    with pytest.raises(ROSConfigError, match="no frame at episode_idx=2"):
        read_frame_trace(root=tmp_path, episode_idx=2, frame_idx=0)


def test_pre_issue_109_dataset_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path, with_trace=False)
    with pytest.raises(ROSConfigError, match="no trace_id/span_id columns"):
        read_frame_trace(root=tmp_path, episode_idx=0, frame_idx=0)
