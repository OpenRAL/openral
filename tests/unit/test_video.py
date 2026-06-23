from __future__ import annotations

import numpy as np
from openral_sim._video import _annotate_panel


def test_annotate_panel_preserves_shape_and_writes_banner() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    out = _annotate_panel(frame, "world render (not a camera)")
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert int(out[:24].sum()) > 0
