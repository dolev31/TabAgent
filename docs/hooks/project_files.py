"""mkdocs hook: the repository's own files as pages of the site.

CHANGELOG.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md and LICENSE live at the repository root, where
GitHub and PyPI read them. The site shows the same files, so there is one copy of each: they are added to the
build as generated pages, their links to each other are rewritten to the site's pages, and "Edit this page"
points at the root file.
"""

from __future__ import annotations

import re
from pathlib import Path

from mkdocs.structure.files import File

#: root file -> (page in the site, title, kind label)
PAGES = {
    "CHANGELOG.md": ("changelog.md", "Changelog"),
    "CONTRIBUTING.md": ("project/contributing.md", "Contributing"),
    "SECURITY.md": ("project/security.md", "Security policy"),
    "CODE_OF_CONDUCT.md": ("project/code-of-conduct.md", "Code of conduct"),
    "LICENSE": ("project/license.md", "License"),
}
_REPO = "https://github.com/dolev31/TabAgent"
_REWRITES = [
    (r"\]\(CODE_OF_CONDUCT\.md\)", "](code-of-conduct.md)"),
    (r"\]\(SECURITY\.md\)", "](security.md)"),
    (r"\]\(docs/reproducing\.md\)", "](../reproducing.md)"),
    (r"\]\(docs/releasing\.md\)", f"]({_REPO}/blob/main/docs/releasing.md)"),
]


def _page(root_name: str, text: str, title: str, page: str) -> str:
    if root_name == "LICENSE":
        notice = "TabAgent is released under the Apache License, Version 2.0."
        body = f"# {title}\n\n{notice}\n\n```text\n{text.strip()}\n```\n"
    else:
        body = text
        for pattern, new in _REWRITES:
            body = re.sub(pattern, new, body)
        if page == "changelog.md":
            body = body.replace("(docs/reproducing.md)", "(reproducing.md)").replace(
                "(../reproducing.md)", "(reproducing.md)"
            )
    kind = "Project" if page.startswith("project/") else "Release notes"
    front = f'---\ntitle: "{title}"\nkind: "{kind}"\n---\n\n'
    return front + body


def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    for name, (page, title) in PAGES.items():
        source = root / name
        if not source.is_file():
            continue
        generated = File.generated(
            config, page, content=_page(name, source.read_text(encoding="utf-8"), title, page)
        )
        generated.edit_uri = f"../{name}"  # "Edit this page" opens the root file, not docs/<page>
        files.append(generated)
    return files
