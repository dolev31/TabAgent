"""Saved heads on the Hugging Face Hub: ``tabagent.load("hf://OWNER/REPO[@REVISION]")``.

A Hub repository holds a saved head exactly as it is on disk: a ``TabAgent`` artifact, or a head saved by
``onboard``. :func:`resolve` downloads the repository snapshot with ``huggingface_hub.snapshot_download``
(cached by the Hub client, so a second load reads the cache) and returns its local directory. Loading then
proceeds as for a local path: every file is checked against the manifest's SHA-256, and ``trust_code`` and
``trust_extractor`` mean the same. Pin ``@REVISION`` (a commit, tag or branch) for a reproducible deployment.
Needs ``pip install "tabagent[hub]"``.
"""

from __future__ import annotations

from pathlib import Path

PREFIX = "hf://"
_HUB_EXTRA = 'loading from the Hugging Face Hub needs the hub extra: pip install "tabagent[hub]"'


def is_hub_uri(path: object) -> bool:
    """Whether ``path`` is an ``hf://`` URI (a string; a ``Path`` would collapse the double slash)."""
    return isinstance(path, str) and path.startswith(PREFIX)


def parse(uri: str) -> tuple[str, str | None]:
    """``(repo_id, revision)`` of ``hf://OWNER/REPO[@REVISION]``; ``revision`` is None for the default
    branch."""
    repo_id, at, revision = uri[len(PREFIX) :].partition("@")
    owner, _, name = repo_id.partition("/")
    if not owner or not name or "/" in name or (at and not revision):
        raise ValueError(f"expected hf://OWNER/REPO or hf://OWNER/REPO@REVISION, got {uri!r}")
    return repo_id, revision or None


def resolve(path: str | Path) -> Path:
    """The local directory of ``path``: an ``hf://`` URI is downloaded from the Hub (or read from its
    cache); any other path is returned as a ``Path``, untouched."""
    if not is_hub_uri(path):
        return Path(path)
    repo_id, revision = parse(path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(_HUB_EXTRA) from e
    return Path(snapshot_download(repo_id=repo_id, revision=revision, repo_type="model"))
