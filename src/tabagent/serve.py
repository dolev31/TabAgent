"""HTTP decision service for a saved head: the drop-in replacement for the LLM component in a live agent.

    tabagent serve ./my_head --port 8077

``POST /decide`` with ``{"prefix": {...}, "candidates": [...], "k": 9}`` returns the shortlist, per-candidate
scores, the head's confidence and whether the gate defers the decision to the original LLM (App. D).
Requires ``pip install "tabagent[serve]"``.
"""

# No `from __future__ import annotations`: FastAPI must see the request model class, not a string.
from typing import Any, Optional


def create_app(head) -> Any:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class DecideRequest(BaseModel):
        prefix: dict[str, Any] = {}
        candidates: list[Any]
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
    import uvicorn

    from tabagent.api import TabAgentHead

    uvicorn.run(create_app(TabAgentHead.load(path, trust_extractor=trust_extractor)), host=host, port=port)
