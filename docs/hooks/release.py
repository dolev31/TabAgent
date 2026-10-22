"""mkdocs hook: the package version, for the announcement banner and the footer.

Read from ``src/tabagent/__init__.py`` (``__version__``) so the banner can never name a version the
package does not have. Exposed as ``config.extra.release`` (not ``extra.version``, which the theme
reserves for the version selector)."""

from __future__ import annotations

import re
from pathlib import Path


def on_config(config):
    init = Path(config.config_file_path).resolve().parent / "src" / "tabagent" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8"), re.M)
    config.extra["release"] = match.group(1) if match else ""
    return config
