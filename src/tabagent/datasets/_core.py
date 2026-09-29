"""Dataset infrastructure: pinned sources, verified downloads, a local cache, and dataset specs.

No benchmark data is shipped with TabAgent. Each dataset is rebuilt from its upstream release at a pinned
revision, and every downloaded file is checked against the SHA-256 recorded in the source manifest
(``tabagent/datasets/manifests/<source>.json``). Files are resolved in this order:

1. ``$TABAGENT_SOURCE_MIRROR/<source id>/<release id>/files/<path>``: an optional local mirror;
2. ``$TABAGENT_CACHE`` (default ``~/.cache/tabagent``) ``/sources/<source id>/<release id>/files/<path>``;
3. a download from the upstream URL, when ``download=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

USER_AGENT = "tabagent-datasets/1"


def cache_root() -> Path:
    return Path(os.environ.get("TABAGENT_CACHE", Path.home() / ".cache" / "tabagent"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None, timeout: int = 300) -> Path:
    """Atomic download with optional SHA-256 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
        urllib.request.urlopen(req, timeout=timeout) as resp,
    ):
        shutil.copyfileobj(resp, tmp)
    tmp_path = Path(tmp.name)
    if sha256 and sha256_file(tmp_path) != sha256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}")
    tmp_path.replace(dest)
    return dest


_fetch = download  # `Source.path` has a `download` flag that would shadow the function


@dataclass(frozen=True)
class Source:
    """An upstream release pinned to one revision."""

    id: str
    benchmark: str
    url: str
    revision: str
    release_id: str
    license: str
    redistribution: str  # redistributable | access_restricted | unresolved
    citation: str
    notes: str = ""

    @property
    def manifest(self) -> dict[str, dict[str, Any]]:
        """``{path: {"url", "sha256", "size_bytes"}}`` for every file of the release TabAgent uses."""
        res = resources.files("tabagent.datasets") / "manifests" / f"{self.id}.json"
        if not res.is_file():
            return {}
        return {f["path"]: f for f in json.loads(res.read_text())["files"]}

    def local_dir(self) -> Path:
        return cache_root() / "sources" / self.id / self.release_id / "files"

    def path(self, rel: str, download: bool = False) -> Path:
        """Resolve one file of the release (mirror, then cache, then download)."""
        mirror = os.environ.get("TABAGENT_SOURCE_MIRROR")
        if mirror:
            p = Path(mirror) / self.id / self.release_id / "files" / rel
            if p.exists():
                return p
        p = self.local_dir() / rel
        if p.exists():
            return p
        entry = self.manifest.get(rel)
        if not download:
            raise FileNotFoundError(
                f"{self.id}: {rel} is not cached. Call with download=True to fetch it from {self.url} "
                f"at revision {self.revision} (license: {self.license})."
            )
        if entry is None or not entry.get("url"):
            raise FileNotFoundError(f"{self.id}: no download URL recorded for {rel}")
        return _fetch(entry["url"], p, entry.get("sha256"))

    def files(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self.manifest if p.startswith(prefix))


@dataclass(frozen=True)
class DatasetInfo:
    """One decision table TabAgent can build from a source."""

    name: str
    source: Source
    build: Callable[..., pd.DataFrame]
    card: str  # bundled card name (tabagent/cards/<card>.yaml)
    folds: str | None  # bundled fold file (tabagent/datasets/folds/<folds>.json), the paper's split
    decision: str
    paper: str = ""
    description: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    requires_license_acceptance: bool = False
