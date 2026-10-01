"""mkdocs hook: expose the landing page's questions to the theme as page.meta["faq"].

The questions are written once, as ``??? question "..."`` blocks in docs/index.md; docs/overrides/main.html
turns them into FAQPage structured data, so the visible answers and the machine-readable ones cannot drift.
"""

from __future__ import annotations

import re
import textwrap

_BLOCK = re.compile(r'^\?\?\?\+? question "(?P<q>[^"]+)"\n\n(?P<a>(?:(?: {4}.*)?\n)+)', re.M)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _plain(markdown: str) -> str:
    text = _LINK.sub(r"\1", textwrap.dedent(markdown))
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.split())


def on_page_markdown(markdown, page, config, files):
    if page.is_homepage:
        page.meta["faq"] = [{"q": m["q"], "a": _plain(m["a"])} for m in _BLOCK.finditer(markdown)]
    return markdown
