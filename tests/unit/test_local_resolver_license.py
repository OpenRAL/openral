"""The local (in-tree) skill resolver enforces the same license gate as the Hub path.

``rSkill.from_pretrained(commercial_use=...)`` blocks non-commercial weights in a
commercial deployment. The local resolver used to drop ``commercial_deployment``,
so an in-tree non-commercial ``vla`` / ``wam`` rSkill ran unchecked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError

pytest.importorskip("rclpy")

_REPO = Path(__file__).resolve().parents[2]


def test_local_resolver_refuses_non_commercial_weights_in_a_commercial_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_rskill_ros.rskill_runner_node import make_local_skill_resolver

    monkeypatch.delenv("OPENRAL_ALLOW_NONCOMMERCIAL", raising=False)
    manifest = (_REPO / "rskills/smolvla-libero/rskill.yaml").read_text(encoding="utf-8")
    skill = tmp_path / "nc-skill"
    skill.mkdir()
    (skill / "rskill.yaml").write_text(
        manifest.replace('license: "apache-2.0"', 'license: "nvidia_non_commercial"'),
        encoding="utf-8",
    )
    resolver = make_local_skill_resolver([tmp_path])
    with pytest.raises(ROSConfigError, match=r"(?i)commercial"):
        resolver(
            rskill_id="OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16",
            revision="",
            prompt="pick",
            prompt_metadata_json="",
            description=None,
            commercial_deployment=True,
        )
