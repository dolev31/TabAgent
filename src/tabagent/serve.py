"""HTTP decision service for a saved head: the drop-in replacement for the LLM component in a live agent.

    tabagent serve ./my_head --port 8077

``POST /decide`` with ``{"prefix": {...}, "candidates": [...], "k": 9}`` returns ``TabAgentHead.decide``'s
answer: the shortlisted ``candidates``, per-candidate ``scores``, the head's ``confidence``, whether the
decision is ``deferred`` to the original LLM (by the confidence gate, App. N.3, in the loop App. O.2, or
because the extractor failed on the prefix) and the extractor's ``error``, if any. ``prefix`` is the
adapter-normalized prefix; send ``"run"`` instead (the run so far in the format of the training logs, with
its run-level task fields) to have the head build it with its recorded adapter. A single-class head takes no
candidates and answers with ``choice``. An invalid request (no candidates for a ranking head, both a prefix
and a run, ...) is answered with status 400. ``GET /health`` and ``GET /card`` describe the head. Requires
``pip install "tabagent[serve]"``; ``RemoteHead`` is its client, standard library only.
"""

# No `from __future__ import annotations`: FastAPI must see the request model class, not a string.
import json
import urllib.error
import urllib.request
from typing import Any, Optional

_SERVE_EXTRA = 'serving a head needs the serve extra: pip install "tabagent[serve]"'


def create_app(head) -> Any:
    """The FastAPI application serving ``head`` (a ``TabAgentHead``)."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    class DecideRequest(BaseModel):
        prefix: Optional[dict[str, Any]] = None  # noqa: UP045
        run: Optional[dict[str, Any]] = None  # noqa: UP045
        candidates: Optional[list[Any]] = None  # noqa: UP045
        k: Optional[int] = None  # noqa: UP045
        defer_below: Optional[float] = None  # noqa: UP045

    app = FastAPI(title="TabAgent decision head", version="1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "target": head.target, "card": head.card.name}

    @app.get("/card")
    def card() -> dict[str, Any]:
        return head.card.to_dict()

    @app.post("/decide")
    def decide(req: DecideRequest) -> dict[str, Any]:
        live = {"run": req.run} if req.run is not None else {}
        prefix = req.prefix if req.prefix is not None or req.run is not None else {}
        try:
            return head.decide(prefix, req.candidates, k=req.k, defer_below=req.defer_below, **live)
        except ValueError as e:  # a request the head cannot answer; extractor failures are deferred instead
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


class RemoteHead:
    """A head served by ``tabagent serve``, called over HTTP with the standard library (no extra needed).

    ``decide`` takes the arguments of ``TabAgentHead.decide`` and returns the service's answer, so an agent
    whose environment cannot host the head (``ToolFilter("http://...")``, say) uses it like a local one. Send
    ``run=`` (the service builds the prefix with the head's recorded adapter) and JSON candidates (tool
    schemas, ids or ``{"id", "description", "args"}`` records). When the service cannot be reached, times
    out or fails (status 5xx), the decision is ``deferred`` with the reason in ``error``, as when a local
    extractor fails; a request the service rejects (status 4xx) raises ``ValueError``.
    """

    def __init__(self, url: str, timeout: float = 10.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def decide(
        self,
        prefix: Optional[dict] = None,  # noqa: UP045
        candidates: Optional[list] = None,  # noqa: UP045
        k: Optional[int] = None,  # noqa: UP045
        defer_below: Optional[float] = None,  # noqa: UP045
        *,
        run: Optional[dict] = None,  # noqa: UP045
    ) -> dict[str, Any]:
        """The service's ``POST /decide`` answer (see ``TabAgentHead.decide``)."""
        body = {"prefix": prefix, "run": run, "candidates": candidates, "k": k, "defer_below": defer_below}
        data = json.dumps({key: v for key, v in body.items() if v is not None}, default=str).encode()
        request = urllib.request.Request(
            self.url + "/decide", data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if 400 <= e.code < 500:
                raise ValueError(f"the head service rejected the request ({e.code}): {detail}") from e
            error = f"the head service failed ({e.code}): {detail}"
        except (OSError, ValueError) as e:  # unreachable, timed out, or an answer that is not JSON
            error = f"the head service could not answer: {type(e).__name__}: {e}"
        return {"candidates": [], "scores": {}, "confidence": None, "deferred": True, "error": error}


def serve(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8077,
    trust_extractor: bool = False,
    sandbox: Any = None,
    redact: Any = None,
    allow_unpinned: bool = False,
) -> None:
    """Load the head saved at ``path`` (a directory or ``hf://OWNER/REPO[@REVISION]``) and serve it until
    interrupted. ``trust_extractor`` runs its extractor in-process instead of in the sandbox (only for an
    extractor you have reviewed); ``sandbox`` (``"subprocess"``, ``"docker"``, ``"docker:<image>"``) and
    ``redact`` (a function or ``"module:function"``, applied to every ``run`` before its prefix is built) are
    described in ``TabAgentHead.load``, as is ``allow_unpinned`` (serve a head whose card pins no hash for its
    extractor, with a warning)."""
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    from tabagent.api import TabAgentHead

    head = TabAgentHead.load(
        path, trust_extractor=trust_extractor, sandbox=sandbox, redact=redact, allow_unpinned=allow_unpinned
    )
    uvicorn.run(create_app(head), host=host, port=port)
