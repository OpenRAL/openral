"""Tests for the interactive REPL startup banner (Claude-Code-style welcome box).

The banner is a white-bordered box that adapts to the terminal width: ``OPENRAL
v<version>`` sits inline in the top border, the white (no-gradient) logo mark and
the OPENRAL wordmark share the top line, the tagline sits beneath them, and a
right-hand cell carries the community links above a divider above the quick-start
commands. We render through a recording :class:`rich.console.Console` and assert on
the exported plain text so the test is independent of terminal/TTY/colour state.
"""

from __future__ import annotations

from openral_cli.main import render_banner
from rich.console import Console


def _plain(version: str = "9.9.9", width: int = 130) -> str:
    """Render the banner to plain (ANSI-stripped) text via a recording console."""
    console = Console(record=True, width=width, force_terminal=False)
    console.print(render_banner(version, width=width))
    return console.export_text()


def test_banner_shows_version_in_top_border() -> None:
    header = _plain(version="1.2.3").splitlines()[0]
    assert "OPENRAL" in header
    assert "v1.2.3" in header


def test_banner_shows_white_logo_and_wordmark() -> None:
    text = _plain()
    assert "█" in text  # solid-block logo + wordmark
    assert "╔" in text  # box-drawing wordmark
    # The legacy gradient shading characters are gone.
    assert "▒" not in text
    assert "▓" not in text


def test_banner_shows_tagline_and_capabilities() -> None:
    text = _plain()
    assert "OpenRAL" in text
    assert "Open Robot Agentic Layer" in text
    assert "embodied AI" in text
    assert "fast policies" in text
    assert "perception" in text


def test_banner_shows_community_links() -> None:
    text = _plain()
    for label in ("Discord", "GitHub", "Hugging Face", "Website"):
        assert label in text, f"missing link label {label!r}"
    for url in (
        "discord.gg/3paXT2bVyB",
        "github.com/OpenRAL/openral",
        "huggingface.co/OpenRAL",
        "openral.com",
    ):
        assert url in text, f"missing link url {url!r}"


def test_banner_shows_quickstart_commands() -> None:
    text = _plain()
    assert "doctor" in text
    assert "rskill search" in text
    assert "help" in text
    assert "exit" in text
    assert "Ctrl-D" in text


def test_banner_fits_terminal_width() -> None:
    # Responsive: the box never overflows the requested width at any size.
    for width in (140, 122, 116, 100, 84, 72):
        text = _plain(width=width)
        for line in text.splitlines():
            assert len(line) <= width, f"line of {len(line)} cols exceeds width {width}"


def test_banner_narrow_layout_keeps_content() -> None:
    # Below the two-column breakpoint the box stacks but keeps every section.
    text = _plain(width=80)
    assert "OPENRAL" in text.splitlines()[0]
    assert "Open Robot Agentic Layer" in text
    assert "Discord" in text
    assert "rskill search" in text


def test_banner_is_content_sized_not_stretched() -> None:
    # On a very wide terminal the box stays compact (content-sized) instead of
    # stretching to fill, so dragging the window narrower has slack before it
    # breaks. The widest line must be far below the 200-col terminal.
    widest = max(len(line) for line in _plain(width=200).splitlines())
    assert 120 < widest < 140


def test_banner_logo_matches_wordmark_height() -> None:
    # The logo mark is the same number of rows as the OPENRAL wordmark so they
    # align on one line; both use only widely-supported block glyphs.
    from openral_cli.main import _LOGO_ART, _WORDMARK_ART

    assert _LOGO_ART.count("\n") == _WORDMARK_ART.count("\n")
    assert set(_LOGO_ART) <= {"█", "▀", "▄", " ", "\n"}
