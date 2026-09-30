"""detect_adapter picks the adapter from the first log files."""

import json

import pytest
from fake_logs import (
    cuga_trajectory,
    langchain_run,
    messages_api_run,
    openai_messages,
    openai_run,
    tau2_results,
)

from tabagent.adapters import CugaAdapter, OpenAIMessagesAdapter, Tau2Adapter, detect_adapter
from tabagent.tabschema.trajectory import GenericJsonAdapter


def _dir(tmp_path, files):
    d = tmp_path / "logs"
    d.mkdir(parents=True)
    for name, content in files.items():
        (d / name).write_text(content if isinstance(content, str) else json.dumps(content))
    return d


@pytest.mark.parametrize(
    "files",
    [
        {"a.json": openai_run(0)},
        {"runs.jsonl": "\n" + json.dumps(openai_run(0)) + "\n" + json.dumps(openai_run(1))},
        {"many.json": [openai_run(0), openai_run(1)]},
        {"bare.json": openai_messages(0)},
        {"a.json": messages_api_run(0)},
        {"a.json": langchain_run(0)},
    ],
)
def test_chat_messages(tmp_path, files):
    assert type(detect_adapter(_dir(tmp_path, files))) is OpenAIMessagesAdapter


def test_tau2_cuga_and_generic_steps(tmp_path):
    tau = _dir(tmp_path / "t", {"r.json": tau2_results("airline", [("0", ["get_user_details"], 1.0)])})
    assert type(detect_adapter(tau)) is Tau2Adapter
    rounds = [("phone", ["phone_search_contacts"], ["phone_search_contacts"])]
    cuga = _dir(
        tmp_path / "c", {"metadata.json": {"task_ids": ["t1"]}, "t1.json": cuga_trajectory("t1", rounds)}
    )
    assert type(detect_adapter(cuga)) is CugaAdapter
    generic = _dir(tmp_path / "g", {"r.json": {"steps": [{"name": "Router", "data": "x"}], "score": 1}})
    assert type(detect_adapter(generic)) is GenericJsonAdapter
    assert detect_adapter(tmp_path / "t" / "logs" / "r.json", granularity="step").target == "NextToolCall"


def test_unknown_format_lists_the_supported_ones(tmp_path):
    d = _dir(tmp_path, {"x.json": {"episodes": []}, "y.jsonl": "not json"})
    with pytest.raises(ValueError, match="Supported formats") as e:
        detect_adapter(d)
    assert "x.json: ['episodes']" in str(e.value) and "Tau2Adapter" in str(e.value)
    with pytest.raises(ValueError, match="no .json or .jsonl files"):
        detect_adapter(_dir(tmp_path / "empty", {}))
    with pytest.raises(FileNotFoundError):
        detect_adapter(tmp_path / "missing")
