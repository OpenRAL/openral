"""openral CLI — the `ros` command entry point."""

from importlib.metadata import version as _pkg_version

from openral_cli.main import app

__all__ = ["app"]
__version__ = _pkg_version("openral-cli")
