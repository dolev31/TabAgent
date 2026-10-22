"""mkdocs hook: one header for every documentation page.

A page that sets ``kind`` in its front matter (Guide, Concept, Reference, Get started, Project) gets,
around its first heading, a small label above the title and, below it, a one-sentence ``lead`` and an
optional ``needs`` line (what you need before you start). The pages stay plain Markdown: the header is
added here, so changing its look is one edit in docs/stylesheets/extra.css."""

from __future__ import annotations

import html
import re

_H1 = re.compile(r"^# .+$", re.M)


def on_page_markdown(markdown, page, config, files):
    meta = page.meta
    kind = meta.get("kind")
    if not kind:
        return markdown
    match = _H1.search(markdown)
    if not match:
        return markdown
    kicker = f'<p class="ta-kicker"><span class="ta-kicker__kind">{html.escape(kind)}</span></p>\n\n'
    lead = meta.get("lead")
    needs = meta.get("needs")
    after = ""
    if lead:
        after += f'\n\n<p class="ta-lead">{html.escape(lead)}</p>'
    if needs:
        after += f'\n\n<p class="ta-needs"><span>You need</span> {html.escape(needs)}</p>'
    return markdown[: match.start()] + kicker + match.group(0) + after + markdown[match.end() :]
