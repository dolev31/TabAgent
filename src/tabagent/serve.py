"""HTTP decision service for a saved head: the drop-in replacement for the LLM component in a live agent.

    tabagent serve ./my_head --port 8077

``POST /decide`` with ``{"prefix": {...}, "candidates": [...], "k": 9}`` returns ``TabAgentHead.decide``'s
answer: the shortlisted ``candidates``, per-candidate ``scores``, the head's ``confidence`` and whether the
confidence gate ``deferred`` the decision to the original LLM (App. N.3; in the loop, App. O.2). A
single-class head takes no candidates and answers with ``choice``. ``GET /health`` and ``GET /card``
describe the head. Requires ``pip install -e ".[serve]"``.
"""

# No `from __future__ import annotations`: FastAPI must see the request model class, not a string.
from typing import Any, Optional

_SERVE_EXTRA = 'serving a head needs the serve extra: pip install -e ".[serve]"'


def create_app(head) -> Any:
    """The FastAPI application serving ``head`` (a ``TabAgentHead``)."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    class DecideRequest(BaseModel):
        prefix: dict[str, Any] = {}
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
        return head.decide(req.prefix, req.candidates, k=req.k, defer_below=req.defer_below)

    return app


def serve(path: str, host: str = "127.0.0.1", port: int = 8077, trust_extractor: bool = False) -> None:
    """Load the head saved at ``path`` and serve it until interrupted. ``trust_extractor`` runs its
    extractor in-process instead of in the sandbox (only for an extractor you have reviewed)."""
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(_SERVE_EXTRA) from e

    from tabagent.api import TabAgentHead

    uvicorn.run(create_app(TabAgentHead.load(path, trust_extractor=trust_extractor)), host=host, port=port)
