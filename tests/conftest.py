"""Shared toy data: a fabricated tool-shortlisting problem (no real agent data); a served head for tests."""

import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from tabagent.table import DecisionSpec, DecisionTable
from tabagent.tabschema import Feature, FeatureCard

TOOLS = {
    "search_flights": "search available flights between two airports",
    "book_flight": "book a seat on a flight for the user",
    "cancel_booking": "cancel an existing booking and refund",
    "get_weather": "get the weather forecast for a city",
    "send_email": "send an email message to a contact",
    "read_inbox": "read the latest email messages in the inbox",
    "create_note": "create a new note with text content",
    "play_song": "play a song or playlist in the music app",
}
INTENTS = [
    (["search_flights", "book_flight"], "find me a flight to {c} and book a seat"),
    (["cancel_booking"], "please cancel my booking to {c} and refund me"),
    (["get_weather"], "what is the weather in {c} tomorrow"),
    (["read_inbox", "send_email"], "read my latest email and reply about {c}"),
    (["create_note"], "make a note about my trip to {c}"),
    (["play_song"], "play some music from {c}"),
]
CITIES = ["paris", "rome", "tokyo", "lima", "oslo", "cairo", "delhi", "quito", "hanoi", "dakar"]

SPEC = DecisionSpec(kind="ranking", group="task", candidate="tool", label="used")


def make_table(n_tasks: int = 60, seed: int = 0) -> DecisionTable:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        used, template = INTENTS[rng.integers(len(INTENTS))]
        text = template.format(c=CITIES[rng.integers(len(CITIES))])
        for tool, desc in TOOLS.items():
            rows.append(
                {
                    "task": f"t{t:03d}",
                    "intent": text,
                    "tool": tool,
                    "tool_desc": desc,
                    "used": int(tool in used),
                }
            )
    return DecisionTable(pd.DataFrame(rows), SPEC)


def make_card(encoder: str = "hashing") -> FeatureCard:
    return FeatureCard(
        name="toy_tools",
        spec=SPEC,
        features=[
            Feature("intent", "state", "text"),
            Feature("tool", "schema", "cat"),
            Feature("tool_desc", "schema", "text"),
            Feature("lex", "schema", "num", "jaccard", {"a": "intent", "b": ["tool", "tool_desc"]}),
            Feature("prior", "dependency", "num", "fitted:candidate_prior", {"fill": "zero"}),
            Feature(
                "sim",
                "dependency",
                "num",
                "fitted:embedding_cosine",
                {"query": "intent", "passage": ["tool", "tool_desc"], "encoder": encoder},
            ),
        ],
        synth_text=["intent"],
        synth_k=2,
        provenance={"kind": "hand-authored", "note": "fabricated test card"},
    )


@pytest.fixture
def table():
    return make_table()


@pytest.fixture
def card():
    return make_card()


@contextmanager
def served(head):
    """``head`` served by ``tabagent serve``'s app on a free local port, for the length of the block: yields
    its URL."""
    uvicorn = pytest.importorskip("uvicorn")
    from tabagent.serve import create_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(
        uvicorn.Config(create_app(head), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the test head server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
