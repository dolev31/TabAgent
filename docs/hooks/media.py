"""mkdocs hook: the landing page's explainer video, played only when its files are there at build time.

The video is not in the repository. The Pages workflow downloads ``tabagent-explainer.mp4``, ``.webm`` and
``.vtt`` from a GitHub Release into docs/assets/video/ before the build (docs/releasing.md).
docs/overrides/home.html plays ``extra.explainer_video`` (the MP4), with the WebM and the captions beside it
when they exist and ``extra.explainer_poster`` as its poster. A relative ``explainer_video`` whose file is
missing is cleared, and the page shows the animated pipeline instead (the static one under reduced motion).
A full URL is kept as it is.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def _local(path: str) -> bool:
    return bool(path) and "://" not in path


def on_config(config):
    extra = config.extra
    docs = Path(config.docs_dir)
    video = extra.get("explainer_video") or ""
    extra["explainer_webm"] = extra["explainer_captions"] = ""
    if _local(video):
        if (docs / video).is_file():
            for key, suffix in (("explainer_webm", ".webm"), ("explainer_captions", ".vtt")):
                sibling = PurePosixPath(video).with_suffix(suffix).as_posix()
                extra[key] = sibling if (docs / sibling).is_file() else ""
        else:
            extra["explainer_video"] = ""
    poster = extra.get("explainer_poster") or ""
    if _local(poster) and not (docs / poster).is_file():
        extra["explainer_poster"] = ""
    return config
