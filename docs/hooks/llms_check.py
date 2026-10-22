"""mkdocs hook: docs/llms.txt may only link to pages the build produced.

llms.txt is written by hand (it is what answer engines read first), so it can drift from the navigation.
After the build, every ``<site_url>...`` link in it must be a page or file in the built site; otherwise
the build warns, which fails ``mkdocs build --strict``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.llms_check")


def on_post_build(config):
    site = Path(config["site_dir"])
    llms = site / "llms.txt"
    base = (config.get("site_url") or "").rstrip("/") + "/"
    if not llms.is_file() or base == "/":
        return
    pages = {p.parent.relative_to(site).as_posix() for p in site.rglob("index.html")}
    for url in re.findall(r"\]\((" + re.escape(base) + r"[^)#\s]*)", llms.read_text(encoding="utf-8")):
        rel = url[len(base) :].strip("/") or "."
        if rel in pages or (site / rel).is_file():
            continue
        log.warning("llms.txt links to %s, which the site does not have", url)
