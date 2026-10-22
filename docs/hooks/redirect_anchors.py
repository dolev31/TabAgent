"""mkdocs hook: send the old pages' section links to the pages that now hold the sections.

The long pages (onboarding, deployment, components) were split into guides and concept pages, so
``/deployment/#production-runbook`` is no longer a section of ``/guides/deploy-and-serve/``. The
redirects plugin forwards the page, and keeps the #fragment as it is. This hook runs after it and
rewrites those redirect pages: the fragment is looked up among every section of the new site (a heading
keeps its slug when it moves) and the visitor lands on the page that holds it. A fragment that is on no
page goes to the redirect's default target."""

from __future__ import annotations

import json
from pathlib import Path

_SECTIONS: dict[str, str] = {}

#: Old page titles whose section was renamed when it became a page of its own: old id -> (page, new id).
_RENAMED = {
    "when-something-goes-wrong": ("guides/when-onboarding-goes-wrong/", "when-onboarding-goes-wrong"),
    "how-the-generated-code-is-verified": ("guides/verify-generated-code/", "verify-the-generated-code"),
}


def _walk(items, url, out):
    for item in items:
        out.setdefault(item.id, url)
        _walk(item.children, url, out)


def on_page_context(context, page, config, nav):
    src = page.file.src_uri
    if not src.startswith(("guides/", "concepts/")) or src.endswith("index.md"):
        return context  # only the pages the old long pages were split into
    found: dict[str, str] = {}
    _walk(page.toc, page.url, found)
    for anchor, url in found.items():
        _SECTIONS.setdefault(anchor, url)
    return context


_PAGE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <meta name="robots" content="noindex">
    <link rel="canonical" href="{default}">
    <script>
    (function () {{
      var base = {base};
      var sections = {sections};
      var renamed = {renamed};
      var hash = window.location.hash.slice(1);
      var dest = base + {default_url};
      if (hash && renamed[hash]) dest = base + renamed[hash][0] + "#" + renamed[hash][1];
      else if (hash && sections[hash]) dest = base + sections[hash] + "#" + hash;
      else if (hash) dest += "#" + hash;
      window.location.replace(dest);
    }})();
    </script>
    <meta http-equiv="refresh" content="0; url={default}">
</head>
<body>
You're being redirected to <a href="{default}">the new page</a>.
</body>
</html>
"""


def on_post_build(config):
    plugin = config.plugins.get("redirects")
    if plugin is None:
        return
    site = Path(config["site_dir"])
    site_url = config.get("site_url") or "/"
    from urllib.parse import urlparse

    base = urlparse(site_url).path or "/"
    base = base if base.endswith("/") else base + "/"
    sections = json.dumps(_SECTIONS, separators=(",", ":"), sort_keys=True)
    for old, new in plugin.redirects.items():
        target = plugin.doc_pages[new.split("#")[0]].url
        old_dir = old[: -len(".md")]
        path = site / old_dir / "index.html"
        if not path.parent.exists():
            continue
        path.write_text(
            _PAGE.format(
                default=f"{site_url.rstrip('/')}/{target}",
                base=json.dumps(base),
                sections=sections,
                renamed=json.dumps({k: list(v) for k, v in _RENAMED.items()}),
                default_url=json.dumps(target),
            ),
            encoding="utf-8",
        )


_CONCEPTS = """<script>
(function () {
  // The old /concepts/ page was split into the pages under it: forward its section links.
  var map = %s;
  var hash = window.location.hash.slice(1);
  if (hash && map[hash]) window.location.replace(%s + map[hash] + "#" + hash);
})();
</script>"""


def on_post_page(output, page, config):
    """The old concepts page lived at the address of the new concepts index: forward its old section links."""
    if page.file.src_uri != "concepts/index.md":
        return output
    from urllib.parse import urlparse

    base = urlparse(config.get("site_url") or "/").path.rstrip("/") + "/"
    old = {
        "one-reduction-for-every-closed-set-decision": "concepts/decision-kinds/",
        "the-decision-table": "concepts/decision-kinds/",
        "leakage-safety-by-construction": "concepts/leakage-safety/",
        "the-protocol": "concepts/leakage-safety/",
    }
    return output.replace("</head>", _CONCEPTS % (json.dumps(old), json.dumps(base)) + "</head>", 1)
