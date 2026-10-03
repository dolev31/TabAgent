"""The production HTTP service on a real saved head: readiness and shutdown of the warm workers, versioning,
structured logs without request content, Prometheus metrics, deadlines that defer instead of failing, batch
decisions and drift monitoring."""

import hashlib
import json
import logging
import os
import re
import time
import warnings

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import tabagent  # noqa: E402
from tabagent import TabAgent, TabHead  # noqa: E402
from tabagent.api import TabAgentHead  # noqa: E402
from tabagent.cli import main  # noqa: E402
from tabagent.drift import DriftMonitor  # noqa: E402
from tabagent.serve import JsonFormatter, create_app, error_class  # noqa: E402
from tabagent.table import DecisionSpec, DecisionTable  # noqa: E402
from tabagent.tabschema import Feature, FeatureCard  # noqa: E402
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, GROUP, LABEL  # noqa: E402
from tabagent.tabschema.warm import WarmSandbox  # noqa: E402

from conftest import CITIES, INTENTS, TOOLS  # noqa: E402

# The extractor reads the prefix's intent; test prefixes can also make it sleep, crash or hang. It imports os,
# signal and time, which the extraction rules forbid: tests that run it pass a sandbox instance (``_pool``),
# which runs what it is given, instead of a sandbox name, which re-applies the rules.
CODE = (
    "import time\n"
    "if 'sleep' in trajectory:\n"
    "    time.sleep(trajectory['sleep'])\n"
    "if 'crash' in trajectory:\n"
    "    import os\n"
    "    os._exit(3)\n"
    "if 'hang' in trajectory:\n"
    "    import signal\n"
    "    signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
    "    while True:\n"
    "        pass\n"
    "features = {'intent': trajectory['intent'], 'n_words': len(trajectory['intent'].split())}\n"
)
TARGET = "ToolShortlist"


def _intent(i: int) -> tuple[str, list[str]]:
    used, template = INTENTS[i % len(INTENTS)]
    return template.format(c=CITIES[i % len(CITIES)]), used


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory):
    """A head saved by TabAgentHead.save: a CatBoost ranking over the toy tools, fed by CODE."""
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    rows = []
    for i in range(60):
        intent, used = _intent(i)
        for tool in TOOLS:
            rows.append(
                {
                    GROUP: f"t{i}",
                    DECISION: f"t{i}#0",
                    "intent": intent,
                    "n_words": len(intent.split()),
                    CANDIDATE: tool,
                    "candidate_name": tool,
                    LABEL: int(tool in used),
                }
            )
    card = FeatureCard(
        "serving_test",
        spec,
        [
            Feature("intent", "state", "text", "extractor"),
            Feature("n_words", "state", "num", "extractor"),
            Feature("candidate_name", "schema", "cat"),
        ],
    )
    table = DecisionTable(pd.DataFrame(rows), spec)
    agent = TabAgent(card, head=TabHead("ranking", iterations=30)).fit(table)
    out = tmp_path_factory.mktemp("heads") / "head"
    TabAgentHead(agent, TARGET, CODE).save(out)
    return out


def _pool(size=1, **kw) -> WarmSandbox:
    return WarmSandbox(size=size, **kw)


def _decide(client, prefix, **kw):
    return client.post("/decide", json={"prefix": prefix, "candidates": list(TOOLS), "k": 2, **kw})


def _wait_ready(client, timeout=30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if client.get("/ready").status_code == 200:
            return True
        time.sleep(0.05)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


_SAMPLE = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{([a-zA-Z_][a-zA-Z0-9_]*="([^"\\]|\\.)*",?)*\})? \S+$')


def _metric(text: str, line_start: str) -> float:
    found = [ln for ln in text.splitlines() if ln.startswith(line_start + " ")]
    assert found, f"no sample {line_start} in /metrics"
    return float(found[0].rsplit(" ", 1)[1])


def _check_exposition(text: str) -> None:
    """Every line is a HELP/TYPE comment or a well-formed sample (Prometheus text format 0.0.4)."""
    for line in text.strip().splitlines():
        assert line.startswith(("# HELP ", "# TYPE ")) or _SAMPLE.match(line), line


def _decision_records(caplog):
    return [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "decision"]


# --------------------------------------------------------------------------- lifecycle
def test_ready_waits_for_the_warm_workers_and_shutdown_stops_them(head_dir):
    head = TabAgentHead.load(head_dir, sandbox=_pool(2))
    with TestClient(create_app(head, warm=False)) as client:  # lazy: workers start with the decisions
        assert client.get("/health").json() == {"status": "ok", "target": TARGET, "card": "serving_test"}
        assert client.get("/ready").status_code == 200 and head.extractor_pool.stats()["live"] == 0

    head = TabAgentHead.load(head_dir, sandbox=_pool(2))  # warm workers, started with the application
    with TestClient(create_app(head, request_timeout=5)) as client:
        assert _wait_ready(client)  # ready with min_ready (1) worker; the others follow
        pool = head.extractor_pool
        end = time.monotonic() + 20
        while client.get("/ready").json()["workers"]["live"] < pool.size and time.monotonic() < end:
            time.sleep(0.02)
        workers = client.get("/ready").json()["workers"]
        assert workers["live"] == workers["size"] == pool.size and workers["restarts"] == 0
        pids = pool.pids()
        assert _decide(client, {"intent": "play some music from oslo"}).json()["error"] is None
    assert pool.stats()["live"] == 0 and not any(_alive(p) for p in pids)  # stopped at shutdown


def test_version_identifies_the_library_and_the_head(head_dir):
    head = TabAgentHead.load(head_dir, sandbox=_pool(2))
    with TestClient(create_app(head, request_timeout=3)) as client:
        v = client.get("/version").json()
    manifest = json.loads((head_dir / "model" / "manifest.json").read_text())
    model_sha = hashlib.sha256(json.dumps(sorted(manifest["sha256"].items())).encode()).hexdigest()
    assert v["tabagent_version"] == tabagent.__version__
    assert v["head"] == {
        "id": model_sha[:12],
        "target": TARGET,
        "card": "serving_test",
        "kind": "ranking",
        "extractor_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
        "model_sha256": model_sha,
        "created_at": manifest["created"],
        "artifact_tabagent_version": tabagent.__version__,
    }
    assert v["serving"] == {"extractor": "warm", "workers": 2, "request_timeout_s": 3, "drift": "on"}


# --------------------------------------------------------------------------- logs and metrics
def test_each_decision_is_logged_and_counted_without_its_content(head_dir, caplog):
    caplog.set_level(logging.INFO, logger="tabagent.serve")
    head = TabAgentHead.load(head_dir, sandbox=_pool())
    with TestClient(create_app(head)) as client:
        r = _decide(client, {"intent": "what is the weather in oslo tomorrow"}, k=1)
        head_id = client.get("/version").json()["head"]["id"]
        text = client.get("/metrics").text
    assert r.status_code == 200 and r.json()["candidates"] == ["get_weather"]
    assert re.fullmatch("[0-9a-f]{32}", r.headers["X-Request-ID"])  # generated when the client sends none
    (rec,) = _decision_records(caplog)
    f = rec.fields
    assert f["request_id"] == r.headers["X-Request-ID"] and f["head_id"] == head_id
    assert set(f["latency_ms"]) == {"total", "prefix", "extraction", "scoring"}
    assert f["latency_ms"]["total"] >= f["latency_ms"]["extraction"] > 0
    assert (f["status"], f["deferred"], f["defer_reason"], f["error_class"]) == (200, False, None, None)
    assert "oslo" not in json.dumps(f) + rec.getMessage()  # the request's content stays out of the logs

    _check_exposition(text)
    assert _metric(text, 'tabagent_decisions_total{endpoint="decide",outcome="head"}') == 1
    assert _metric(text, 'tabagent_http_requests_total{endpoint="/decide",code="200"}') == 1
    for stage in ("total", "prefix", "extraction", "scoring"):
        assert _metric(text, f'tabagent_decision_seconds_count{{stage="{stage}"}}') == 1
        assert _metric(text, f'tabagent_decision_seconds_bucket{{stage="{stage}",le="+Inf"}}') == 1
    assert _metric(text, 'tabagent_extractor_workers{state="size"}') == 1
    assert _metric(text, "tabagent_extractor_restarts_total") == 0
    assert _metric(text, "tabagent_ready") == 1
    assert f'id="{head_id}"' in text and "tabagent_drift_window_decisions 1" in text


def test_request_ids_are_echoed_and_content_is_logged_only_on_request(head_dir, caplog):
    caplog.set_level(logging.INFO, logger="tabagent.serve")
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    with TestClient(create_app(head, log_content=True)) as client:
        r = client.post(
            "/decide",
            json={"prefix": {"intent": "play some music from lima"}, "candidates": list(TOOLS), "k": 1},
            headers={"X-Request-ID": "agent-run-7"},
        )
    assert r.headers["X-Request-ID"] == "agent-run-7"
    (rec,) = _decision_records(caplog)
    assert rec.fields["request_id"] == "agent-run-7"
    assert rec.fields["request"]["prefix"] == {"intent": "play some music from lima"}
    assert rec.fields["answer"]["candidates"] == r.json()["candidates"]
    line = json.loads(JsonFormatter().format(rec))  # one JSON object per record
    assert line["level"] == "INFO" and line["request_id"] == "agent-run-7"
    assert line["logger"] == "tabagent.serve"


# --------------------------------------------------------------------------- deadlines and failures
@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_slow_crashing_and_hanging_extractors_are_deferred_and_the_workers_replaced(head_dir, caplog):
    caplog.set_level(logging.INFO, logger="tabagent.serve")
    head = TabAgentHead.load(head_dir, sandbox=_pool())
    with TestClient(create_app(head, request_timeout=1)) as client:
        assert _wait_ready(client)
        t0 = time.monotonic()
        slow = _decide(client, {"intent": "book a flight to rome", "sleep": 20}).json()
        assert time.monotonic() - t0 < 5
        crash = _decide(client, {"intent": "book a flight to rome", "crash": 1}).json()
        hang = _decide(client, {"intent": "book a flight to rome", "hang": 1}).json()
        fine = _decide(client, {"intent": "book a flight to rome"}).json()
        text = client.get("/metrics").text
    for answer in (slow, crash, hang):  # deferred to the LLM, never an error status; still a ranking
        assert answer["deferred"] is True and len(answer["candidates"]) == 2
    assert "timed out" in slow["error"] and "exited early" in crash["error"] and "killed" in hang["error"]
    assert fine["error"] is None and fine["deferred"] is False
    assert _metric(text, 'tabagent_deferrals_total{reason="extractor_timeout"}') == 2
    assert _metric(text, 'tabagent_deferrals_total{reason="extractor_crash"}') == 1
    assert _metric(text, 'tabagent_errors_total{class="extractor_timeout"}') == 2
    assert _metric(text, "tabagent_extractor_restarts_total") == 2  # the crash and the killed hang
    assert _metric(text, 'tabagent_decisions_total{endpoint="decide",outcome="deferred"}') == 3
    classes = [r.fields["error_class"] for r in _decision_records(caplog)]
    assert classes == ["extractor_timeout", "extractor_crash", "extractor_timeout", None]
    assert [r.fields["error_type"] for r in _decision_records(caplog)][:2] == ["TimeoutError", None]


def test_a_decision_past_its_deadline_is_answered_deferred(head_dir):
    head = TabAgentHead.load(head_dir, trust_extractor=True)  # in-process code cannot be interrupted
    with TestClient(create_app(head, request_timeout=0.2)) as client:
        t0 = time.monotonic()
        r = _decide(client, {"intent": "book a flight to rome", "sleep": 4})
        elapsed = time.monotonic() - t0
        text = client.get("/metrics").text
    body = r.json()
    assert r.status_code == 200 and elapsed < 3.5
    assert body["deferred"] is True and body["confidence"] is None and body["scores"] == {}
    assert body["candidates"] == list(TOOLS)  # the response's shape: every candidate, in input order
    assert body["error"].startswith("TimeoutError")
    assert _metric(text, 'tabagent_deferrals_total{reason="request_timeout"}') == 1


def test_invalid_requests_are_400_and_batches_answer_each_request(head_dir):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    with TestClient(create_app(head, max_batch=3)) as client:
        assert _decide(client, {"intent": "x"}, candidates=None).status_code == 400
        batch = [
            {"prefix": {"intent": "play some music from rome"}, "candidates": list(TOOLS), "k": 1},
            {"prefix": {"intent": "play some music from rome"}},  # no candidates
            {"run": {}, "prefix": {"intent": "x"}, "candidates": list(TOOLS)},  # both
        ]
        r = client.post("/decide_batch", json={"requests": batch})
        too_many = client.post("/decide_batch", json={"requests": batch * 2})
        text = client.get("/metrics").text
    results = r.json()["results"]
    assert r.status_code == 200 and results[0]["candidates"] == ["play_song"]
    assert results[1]["status"] == 400 and "candidates" in results[1]["detail"]
    assert results[2]["status"] == 400 and "not both" in results[2]["detail"]
    assert too_many.status_code == 400 and "at most 3" in too_many.json()["detail"]
    assert _metric(text, 'tabagent_errors_total{class="bad_request"}') == 3
    assert _metric(text, 'tabagent_decisions_total{endpoint="decide_batch",outcome="head"}') == 1


# --------------------------------------------------------------------------- drift
@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_drift_is_reported_exported_and_logged(head_dir, caplog):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    monitor = DriftMonitor(head.agent.feature_stats_, window=40, min_decisions=20, check_every=10)
    caplog.set_level(logging.INFO, logger="tabagent.drift")
    with TestClient(create_app(head, drift=monitor)) as client:
        for i in range(30):  # decisions like the training ones
            assert _decide(client, {"intent": _intent(i)[0]}).status_code == 200
        before = client.get("/drift").json()
        for i in range(40):  # the agent changed: its prefixes no longer carry the intent
            assert _decide(client, {"goal": _intent(i)[0]}).json()["deferred"] is True
        after = client.get("/drift").json()
        text = client.get("/metrics").text
    assert before["status"] == "ok" and before["drifting"] == []
    assert after["status"] == "drift" and {"intent", "n_words"} <= set(after["drifting"])
    assert after["features"]["intent"]["reasons"] == ["null_rate"]
    assert after["features"]["intent"]["null_rate_shift"] == pytest.approx(1.0)
    assert _metric(text, 'tabagent_feature_null_rate_shift{feature="intent"}') == pytest.approx(1.0)
    assert _metric(text, 'tabagent_feature_drifting{feature="intent"}') == 1
    assert _metric(text, "tabagent_drift_features_drifting") >= 2
    assert _metric(text, "tabagent_drift_window_decisions") == 40
    warned = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "feature_drift"]
    assert {r.fields["feature"] for r in warned} >= {"intent", "n_words"}


def test_drift_without_training_statistics_is_reported_unavailable(head_dir):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    head.agent.feature_stats_ = None  # a head saved before tabagent recorded them
    with TestClient(create_app(head)) as client:
        rep = client.get("/drift").json()
        text = client.get("/metrics").text
    assert rep["status"] == "unavailable" and "refit" in rep["reason"]
    assert "tabagent_feature_psi" not in text


# --------------------------------------------------------------------------- helpers and the CLI
def test_error_classes():
    assert error_class(None) is None
    # by the failure's kind, which TabAgentHead.decide reports in its trace
    kinds = {"error": "extractor_error", "timeout": "extractor_timeout", "crash": "extractor_crash"}
    kinds |= {
        "broken": "extractor_broken",
        "unavailable": "sandbox_unavailable",
        "no_extractor": "no_extractor",
    }
    for kind, cls in kinds.items():
        assert error_class("KeyError: 'timed out'", kind) == cls  # the message does not matter
    # from the message, for a head that reports no kind
    assert error_class("TimeoutError: timed out after 1s on this trajectory") == "extractor_timeout"
    assert error_class("the sandboxed interpreter exited early (exit status 3)") == "extractor_crash"
    assert error_class("KeyError: 'instruction'. The live prefix has no 'instruction' field") == (
        "extractor_error"
    )


def test_the_cli_passes_the_serving_options(monkeypatch, head_dir):
    import tabagent.serve as serve_module

    seen = {}
    monkeypatch.setattr(serve_module, "serve", lambda *a, **kw: seen.update(args=a, kw=kw))
    argv = ["serve", str(head_dir), "--workers", "3", "--request-timeout", "2.5", "--log-content"]
    assert main([*argv, "--no-drift", "--log-format", "text", "--threads", "3"]) == 0
    kw = seen["kw"]
    assert seen["args"][3] is False and kw["sandbox"] == "warm" and kw["workers"] == 3
    assert kw["request_timeout"] == 2.5 and kw["log_content"] and kw["drift"] is False
    assert kw["log_format"] == "text" and kw["threads"] == 3
    assert main(["serve", str(head_dir), "--request-timeout", "0"]) == 0
    assert seen["kw"]["request_timeout"] is None


def test_serve_builds_a_sized_warm_pool_under_the_request_timeout(monkeypatch, head_dir):
    uvicorn = pytest.importorskip("uvicorn")  # the serve extra; CI's test job installs FastAPI only

    import tabagent.serve as serve_module

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, kw=kw))
    monkeypatch.setattr(serve_module, "configure_logging", lambda fmt: None)
    with warnings.catch_warnings():
        serve_module.serve(str(head_dir), workers=3, request_timeout=2.5)
    svc = seen["app"].state.service
    assert svc.pool.size == 3 and svc.pool.timeout_s == 2.5
    assert seen["kw"]["timeout_graceful_shutdown"] == 30.0 and seen["kw"]["access_log"] is False
    with pytest.raises(ValueError, match="workers="):
        serve_module.serve(str(head_dir), sandbox="docker", workers=2)


def test_the_built_in_exporter_writes_the_prometheus_text_format():
    from tabagent.prometheus import Registry

    r = Registry(include_default=False)
    c = r.counter("demo_requests_total", "Requests.\nSecond line.", ("path",))
    c.inc(path='/a"b\\c\nd')
    c.inc(2, path="/x")
    h = r.histogram("demo_seconds", "Latency.", (), (0.1, 1))
    for v in (0.05, 0.5, 5):
        h.observe(v)
    r.gauge("demo_up", "Up.", (), lambda: 1)
    r.gauge("demo_skipped", "No value yet.", (), lambda: None)
    text = r.render()
    _check_exposition(text)
    assert "# HELP demo_requests_total Requests.\\nSecond line." in text
    assert 'demo_requests_total{path="/a\\"b\\\\c\\nd"} 1.0' in text
    assert 'demo_requests_total{path="/x"} 2.0' in text
    assert 'demo_seconds_bucket{le="0.1"} 1' in text and 'demo_seconds_bucket{le="1.0"} 2' in text
    assert 'demo_seconds_bucket{le="+Inf"} 3' in text and "demo_seconds_sum 5.55" in text
    assert "demo_seconds_count 3" in text and "demo_up 1" in text and "\ndemo_skipped " not in text
    with pytest.raises(ValueError, match="takes labels"):
        c.inc(other="x")
    with pytest.raises(ValueError, match="already registered"):
        r.counter("demo_requests_total", "again")


def test_json_logging_is_configured_for_the_service_and_the_library(capsys):
    import tabagent.serve as serve_module

    root, lib = logging.getLogger(), logging.getLogger("tabagent")
    saved = root.handlers[:], root.level, lib.handlers[:], lib.level
    try:
        lib.setLevel(logging.WARNING)  # as the CLI leaves it without -v
        serve_module.configure_logging("json")
        logging.getLogger("tabagent.serve").info("hello", extra={"fields": {"event": "x", "n": 1}})
        logging.getLogger("tabagent.api").warning("TabSchema extractor failed on a live prefix; secret data")
        data = {"intent": "secret value"}
        try:
            raise KeyError(data["intent"])
        except KeyError:
            logging.getLogger("tabagent.serve").exception("boom")
    finally:
        root.handlers[:], lib.handlers[:] = saved[0], saved[2]
        root.setLevel(saved[1])
        lib.setLevel(saved[3])
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    assert [(r["message"], r.get("event")) for r in lines] == [("hello", "x"), ("boom", None)]
    assert lines[0]["n"] == 1 and lines[0]["logger"] == "tabagent.serve"
    assert lines[1]["exception"] == "KeyError" and "secret" not in json.dumps(lines)


# --------------------------------------------------------------------------- review fixes
@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_an_extractor_that_breaks_the_rules_is_not_ready_and_says_why(head_dir, caplog):
    caplog.set_level(logging.INFO, logger="tabagent")
    head = TabAgentHead.load(head_dir)  # a sandbox name: the rules apply, and CODE imports os
    with TestClient(create_app(head, request_timeout=2)) as client:
        time.sleep(0.3)
        ready = client.get("/ready")
        answer = _decide(client, {"intent": "play some music from oslo"}).json()
        text = client.get("/metrics").text
    assert ready.status_code == 503 and "cannot run" in ready.json()["reason"]
    assert answer["deferred"] and "violates the extraction rules" in answer["error"]
    assert _metric(text, "tabagent_extractor_broken") == 1 and _metric(text, "tabagent_ready") == 0
    assert _metric(text, 'tabagent_deferrals_total{reason="extractor_broken"}') == 1
    failed = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "workers_failed"]
    assert failed and failed[0].levelname == "ERROR"
    assert not any(getattr(r, "fields", {}).get("event") == "workers_started" for r in caplog.records)


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_an_extractor_that_does_not_compile_is_reported_not_restarted(head_dir):
    agent = TabAgentHead.load(head_dir).agent
    head = TabAgentHead(agent, TARGET, "features = {'intent': trajectory['intent']\n", sandbox=_pool(2))
    with TestClient(create_app(head, request_timeout=2)) as client:
        time.sleep(0.5)
        ready = client.get("/ready")
        answers = [_decide(client, {"intent": "x"}).json() for _ in range(5)]
        text = client.get("/metrics").text
        pool = head.extractor_pool.stats()
    assert ready.status_code == 503 and "SyntaxError" in ready.json()["reason"]
    assert all(a["deferred"] and "SyntaxError" in a["error"] for a in answers)
    assert _metric(text, 'tabagent_deferrals_total{reason="extractor_broken"}') == 5
    assert _metric(text, "tabagent_extractor_start_failures_total") >= 1
    assert pool["started"] == 0  # no interpreter per request


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_a_batch_has_one_deadline(head_dir):
    head = TabAgentHead.load(head_dir, sandbox=_pool(2))
    hang = {"prefix": {"intent": "book a flight", "hang": 1}, "candidates": list(TOOLS), "k": 1}
    with TestClient(create_app(head, request_timeout=0.5)) as client:
        assert _wait_ready(client)
        t0 = time.monotonic()
        r = client.post("/decide_batch", json={"requests": [hang] * 16})
        elapsed = time.monotonic() - t0
    results = r.json()["results"]
    assert r.status_code == 200 and elapsed < 5  # not 16 items x their own deadline
    assert all(res["deferred"] for res in results) and len(results) == 16
    late = [res for res in results if res["error"].startswith("TimeoutError: no answer")]
    assert late and all(res["candidates"] == list(TOOLS) and res["scores"] == {} for res in late)


def test_status_codes_separate_bad_requests_from_head_failures(head_dir):
    class Broken:  # a head with a bug: a plain ValueError is not the client's fault
        target, card = "x", type("C", (), {"name": "broken", "to_dict": lambda self: {}})()

        def decide(self, prefix, candidates, k=None, defer_below=None):
            raise ValueError("a bug in the head")

    head = TabAgentHead.load(head_dir, trust_extractor=True)
    with TestClient(create_app(head)) as client:
        nameless = _decide(client, {"intent": "x"}, candidates=[{"description": "no name"}])
        schema = client.post("/decide", json={"prefix": {"intent": "x"}, "candidates": "not a list"})
    with TestClient(create_app(Broken())) as client:
        bug = client.post("/decide", json={"prefix": {}, "candidates": ["a"]})
    assert nameless.status_code == 400 and "candidate" in nameless.json()["detail"]
    assert schema.status_code == 422
    assert bug.status_code == 500 and "ValueError" in bug.json()["detail"]


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_request_timeouts_must_be_positive_and_finite(head_dir, bad):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    with pytest.raises(ValueError, match="positive"):
        create_app(head, request_timeout=bad)


# --------------------------------------------------------------------------- final review
@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_allow_unchecked_extractor_runs_a_refused_extractor_in_the_sandbox_with_a_warning(head_dir, caplog):
    """An older head whose extract.py imports time and os was served before the rules were re-applied at serve
    time; the opt-out keeps the sandbox but skips the check, loudly."""
    refused = TabAgentHead.load(head_dir)
    assert "violates the extraction rules" in refused.decide({"intent": "x"}, list(TOOLS), k=1)["error"]
    caplog.set_level(logging.WARNING, logger="tabagent.api")
    with pytest.warns(UserWarning, match="allow_unchecked_extractor=True.*import: time"):
        head = TabAgentHead.load(head_dir, allow_unchecked_extractor=True)
        answer = head.decide({"intent": "play some music from oslo"}, list(TOOLS), k=1)
    assert answer["error"] is None and answer["candidates"] == ["play_song"]
    assert head.extractor_pool is not None  # still in the warm sandbox
    assert any(getattr(r, "fields", {}).get("event") == "unchecked_extractor" for r in caplog.records)
    head.close()


def test_the_cli_and_serve_pass_the_unchecked_opt_out(monkeypatch, head_dir):
    import uvicorn

    import tabagent.serve as serve_module

    seen = {}
    monkeypatch.setattr(serve_module, "serve", lambda *a, **kw: seen.update(kw=kw))
    assert main(["serve", str(head_dir), "--allow-unchecked-extractor"]) == 0
    assert seen["kw"]["allow_unchecked_extractor"] is True
    monkeypatch.undo()
    apps = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.update(app=app))
    monkeypatch.setattr(serve_module, "configure_logging", lambda fmt: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        serve_module.serve(str(head_dir), allow_unchecked_extractor=True)
    assert apps["app"].state.service.head.allow_unchecked_extractor


def test_k_must_be_positive(head_dir):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    with TestClient(create_app(head)) as client:
        codes = [_decide(client, {"intent": "x"}, k=k).status_code for k in (-1, 0, 1)]
    assert codes == [400, 400, 200]


def test_an_unreachable_service_defers_in_the_service_shape():
    from tabagent.serve import RemoteHead

    answer = RemoteHead("http://127.0.0.1:9", timeout=2).decide({"intent": "x"}, list(TOOLS), k=2)
    assert answer["deferred"] and answer["candidates"] == list(TOOLS) and answer["scores"] == {}
    assert "could not answer" in answer["error"]


def test_metrics_read_the_latest_drift_evaluation_and_the_decision_queue(head_dir):
    head = TabAgentHead.load(head_dir, trust_extractor=True)
    monitor = DriftMonitor(head.agent.feature_stats_, window=40, min_decisions=20, check_every=10)
    with TestClient(create_app(head, drift=monitor)) as client:
        for i in range(20):
            _decide(client, {"intent": _intent(i)[0]})
        monitor.report()  # the latest evaluation
        evaluations = []
        real = monitor._evaluate
        monitor._evaluate = lambda: evaluations.append(1) or real()
        for _ in range(3):
            text = client.get("/metrics").text
    assert evaluations == []  # scrapes do not re-evaluate the window
    assert _metric(text, "tabagent_drift_window_decisions") == 20
    assert _metric(text, "tabagent_queued_decisions") == 0
    assert _metric(text, "tabagent_inflight_decisions") == 0
