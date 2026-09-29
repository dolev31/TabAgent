import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tabagent.serve import create_app  # noqa: E402


class FakeHead:
    target = "ShortlisterAgent"

    class card:  # noqa: N801
        name = "demo"

        @staticmethod
        def to_dict():
            return {"name": "demo"}

    def decide(self, prefix, candidates, k=None, defer_below=None):
        return {"choice": candidates[:k], "scores": {}, "confidence": 1.0, "defer": False}


def test_decide_endpoint():
    client = TestClient(create_app(FakeHead()))
    assert client.get("/health").json()["target"] == "ShortlisterAgent"
    r = client.post("/decide", json={"prefix": {"steps": []}, "candidates": ["a", "b", "c"], "k": 2})
    assert r.status_code == 200 and r.json()["choice"] == ["a", "b"]
