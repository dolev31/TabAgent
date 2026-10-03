"""Live decisions of an onboarded head: train/serve parity of the prefix, deferral instead of a crash when the
extractor fails, the HTTP service and the tool filter, sandbox selection and redaction at serve time."""

import json
import warnings

import pytest
from fake_logs import messages_api_tools, openai_run, openai_tools, task, write_openai_logs
from test_adapter_discovery import responder

import tabagent
from tabagent import TabAgent, TabHead
from tabagent.adapters import TOOL_SHORTLIST, OpenAIMessagesAdapter, describe_adapter, parse_catalog
from tabagent.api import TabAgentHead
from tabagent.cli import main
from tabagent.integrations.openai_tools import ToolFilter
from tabagent.table import DecisionSpec
from tabagent.tabschema import Feature, FeatureCard, TabSchemaConfig
from tabagent.tabschema import llm as llm_module
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, GROUP, LABEL, rows_to_table
from tabagent.tabschema.sandbox import DockerSandbox, ExecResult, SubprocessSandbox
from tabagent.tabschema.warm import WarmSandbox

# An extractor that reads a run-level task field, as TabSchema's often do.
CODE = """
def extract_features_for_classification(trajectory_data):
    users = [s['content'] for s in trajectory_data['steps'] if s['name'] == 'user']
    return {"goal": trajectory_data['instruction'], "first_user": users[0] if users else ""}
features = extract_features_for_classification(trajectory)
"""


def _runs(n=24):
    return [openai_run(i, {"reward": 1.0}, instruction=f"goal: {task(i)[0]}") for i in range(n)]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A head fitted on decision points built by OpenAIMessagesAdapter, with CODE as its extractor."""
    d = tmp_path_factory.mktemp("logs")
    for i, run in enumerate(_runs()):
        (d / f"run{i:02d}.json").write_text(json.dumps(run))
    adapter = OpenAIMessagesAdapter()
    trajs = adapter.load(d)
    dps = [dp for t in trajs for dp in adapter.decision_points(t, TOOL_SHORTLIST)]
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    card = FeatureCard(
        "live_test",
        spec,
        [
            Feature("goal", "state", "text", "extractor"),
            Feature("first_user", "state", "text", "extractor"),
            Feature("candidate_name", "schema", "cat"),
            Feature("candidate_description", "schema", "text"),
            Feature("lex", "schema", "num", "jaccard", {"a": "goal", "b": ["candidate_name"]}),
        ],
    )
    feats = ExtractorRunner(CODE, trust_extractor=True).run([dp.prefix for dp in dps])
    table = rows_to_table(card, dps, feats, adapter)
    agent = TabAgent(card, head=TabHead("ranking", iterations=30)).fit(table)
    return agent, adapter, trajs, dps


def _head(trained, **kw):
    agent, adapter, _, _ = trained
    kw.setdefault("trust_extractor", True)
    return TabAgentHead(agent, TOOL_SHORTLIST, CODE, adapter=adapter, **kw)


def _live(run, dp):
    """The run as the agent holds it at the decision: the messages before the first tool call."""
    live = {k: v for k, v in run.items() if k not in ("reward", "messages")}
    return {**live, "messages": run["messages"][: dp.step_index]}


def test_training_and_live_prefixes_give_identical_features(trained):
    _, adapter, trajs, dps = trained
    head = _head(trained)
    run, dp = trajs[3].raw, dps[3]
    live = _live(run, dp)
    assert "task_id" not in dp.prefix  # the group id is not a feature (see OpenAIMessagesAdapter)

    assert head.live_prefix(live) == dp.prefix
    assert head.features_for(head.live_prefix(live)) == head.features_for(dp.prefix)

    # the tool filter builds the same prefix from the messages, the tools and the run-level task fields
    tf = ToolFilter(head, k=2, task={"instruction": run["instruction"]})
    prefix = tf.prefix(live["messages"], run["tools"])
    assert prefix == dp.prefix
    assert head.features_for(prefix) == ExtractorRunner(CODE, trust_extractor=True).run([dp.prefix])[0]
    kept = tf(live["messages"], run["tools"])
    assert tf.last["error"] is None and tf.last["deferred"] is False and len(kept) == 2
    tools = list(parse_catalog(run["tools"]).values())
    assert head.decide(run=live, candidates=tools, k=2)["candidates"] == tf.last["candidates"]


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
@pytest.mark.parametrize("trust", [True, False])
def test_an_extractor_failure_defers_the_decision_instead_of_crashing(trained, trust):
    head = _head(trained, trust_extractor=trust)
    prefix = {"steps": [{"name": "user", "content": "book a flight", "tool_calls": [], "tool_name": ""}]}
    out = head.decide(prefix, [t["function"]["name"] for t in openai_tools()], k=2)
    assert out["deferred"] is True and len(out["candidates"]) == 2 and set(out["scores"])
    assert "KeyError" in out["error"] and "'instruction'" in out["error"] and "run-level" in out["error"]
    assert head.features_for(prefix) == {"goal": None, "first_user": None}  # None-filled, as in training


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_the_tool_filter_sends_every_tool_when_the_prefix_lacks_a_training_field(trained):
    _, _, trajs, dps = trained
    live = _live(trajs[0].raw, dps[0])
    tf = ToolFilter(_head(trained), k=2)  # no run-level task fields: 'instruction' is missing
    kept = tf(live["messages"], openai_tools())
    assert len(kept) == len(openai_tools()) and tf.last["deferred"] and "instruction" in tf.last["error"]


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_http_service_defers_and_builds_prefixes_from_runs(trained):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tabagent.serve import create_app

    _, _, trajs, dps = trained
    client = TestClient(create_app(_head(trained)))
    names = [t["function"]["name"] for t in openai_tools()]
    r = client.post("/decide", json={"prefix": {"steps": []}, "candidates": names, "k": 2})
    assert r.status_code == 200 and r.json()["deferred"] is True and "instruction" in r.json()["error"]

    r = client.post("/decide", json={"run": _live(trajs[1].raw, dps[1]), "candidates": names, "k": 2})
    assert r.status_code == 200 and r.json()["error"] is None and len(r.json()["candidates"]) == 2

    r = client.post("/decide", json={"prefix": dps[1].prefix, "candidates": None})
    assert r.status_code == 400 and "candidates" in r.json()["detail"]
    r = client.post("/decide", json={"prefix": dps[1].prefix, "run": {}, "candidates": names})
    assert r.status_code == 400


def test_head_json_records_the_adapter_and_redaction(trained, tmp_path):
    head = _head(trained, redact="copy:deepcopy")
    head.save(tmp_path / "head")
    meta = json.loads((tmp_path / "head" / "head.json").read_text())
    assert meta["adapter"] == describe_adapter(OpenAIMessagesAdapter())
    assert meta["redact"] == "copy:deepcopy"

    with pytest.warns(UserWarning, match="copy:deepcopy"):
        loaded = TabAgentHead.load(tmp_path / "head", trust_extractor=True)
    _, _, trajs, dps = trained
    assert loaded.live_prefix(_live(trajs[2].raw, dps[2])) == dps[2].prefix  # the adapter is rebuilt
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = TabAgentHead.load(tmp_path / "head", trust_extractor=True, redact="copy:deepcopy")

    seen = []

    def scrub(run):
        seen.append(run)
        return {**run, "instruction": "[redacted]"}

    again.redact = scrub
    assert again.live_prefix(_live(trajs[2].raw, dps[2]))["instruction"] == "[redacted]" and seen


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_sandbox_selection(trained):
    head = _head(trained, trust_extractor=False)
    assert isinstance(head._sandbox(), WarmSandbox)  # persistent workers by default
    cold = _head(trained, trust_extractor=False, sandbox="subprocess")._sandbox()
    assert type(cold) is SubprocessSandbox  # a fresh interpreter per decision, on request
    assert _head(trained, trust_extractor=False, sandbox="warm:2")._sandbox().size == 2
    with pytest.raises(ValueError, match="positive number"):
        _head(trained, trust_extractor=False, sandbox="warm:0")._sandbox()
    assert isinstance(_head(trained, trust_extractor=False, sandbox="docker")._sandbox(), DockerSandbox)
    img = _head(trained, trust_extractor=False, sandbox="docker:python:3.11-slim")._sandbox()
    assert isinstance(img, DockerSandbox) and img.image == "python:3.11-slim"
    with pytest.raises(ValueError, match="in-process"):
        _head(trained, trust_extractor=True, sandbox="docker")

    class Recording(SubprocessSandbox):
        calls = 0

        def run_batch(self, code, trajectories):
            Recording.calls += 1
            return super().run_batch(code, trajectories)

    _, _, _, dps = trained
    custom = _head(trained, trust_extractor=False, sandbox=Recording())
    out = custom.decide(dps[0].prefix, ["search_flights", "book_flight", "get_weather"], k=1)
    assert Recording.calls == 1 and out["error"] is None

    class NoDocker(SubprocessSandbox):  # what DockerSandbox does on a machine without Docker
        def run_batch(self, code, trajectories):
            raise FileNotFoundError("docker")

        def run(self, code, trajectory):
            return ExecResult(False, error="FileNotFoundError: docker")

    broken = _head(trained, trust_extractor=False, sandbox=NoDocker())
    out = broken.decide(dps[0].prefix, ["search_flights", "book_flight"], k=1)
    assert out["deferred"] is True and "docker" in out["error"]


def test_cli_onboarded_head_records_its_adapter_and_redaction(tmp_path, monkeypatch):
    """`tabagent onboard --redact` records the adapter and the redaction import path in head.json."""
    logs = write_openai_logs(tmp_path / "logs", n=24)
    monkeypatch.setattr(llm_module, "auto_llm", lambda model=None: llm_module.ScriptedLLM(responder))
    onboard, cfg = tabagent.onboard, TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False)
    quick = {"n_analyze": None, "config": cfg, "iterations": 20}  # offline and fast
    monkeypatch.setattr(tabagent, "onboard", lambda *a, **kw: onboard(*a, **{**kw, **quick}))
    out = tmp_path / "head"
    run = ["--run-dir", str(tmp_path / "run")]
    assert main(["onboard", str(logs), "--redact", "copy:deepcopy", "--out", str(out), *run]) == 0

    meta = json.loads((out / "head.json").read_text())
    assert meta["adapter"]["class"] == "OpenAIMessagesAdapter" and meta["redact"] == "copy:deepcopy"
    with pytest.warns(UserWarning, match="redact"):
        head = tabagent.load(out)
    assert head.adapter["class"] == "OpenAIMessagesAdapter"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        tabagent.load(out, redact="copy:deepcopy")


# ------------------------------------------------------------------------- candidates and text at serve time
@pytest.fixture(scope="module")
def rich(trained):
    """A head whose card reads the candidates' argument hints and embeds text (hashing encoder), as TabSchema
    cards do (candidate_args, context_candidate_sim)."""
    _, adapter, trajs, dps = trained
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    card = FeatureCard(
        "live_rich",
        spec,
        [
            Feature("goal", "state", "text", "extractor"),
            Feature("first_user", "state", "text", "extractor"),
            Feature("candidate_name", "schema", "cat"),
            Feature("candidate_description", "schema", "text"),
            Feature("candidate_args", "schema", "text"),
            Feature(
                "sim",
                "dependency",
                "num",
                "fitted:embedding_cosine",
                {"query": ["first_user"], "passage": ["candidate_description"], "encoder": "hashing"},
            ),
        ],
    )
    feats = ExtractorRunner(CODE, trust_extractor=True).run([dp.prefix for dp in dps])
    table = rows_to_table(card, dps, feats, adapter)
    agent = TabAgent(card, head=TabHead("ranking", iterations=30)).fit(table)
    return TabAgentHead(agent, TOOL_SHORTLIST, CODE, adapter=adapter, trust_extractor=True), table, dps


def _mcp_tools():
    return [
        {"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
        for t in messages_api_tools()
    ]


def test_tool_schemas_as_candidates_score_as_in_training(rich):
    """A scaffold sends its tool list as it holds it: every schema dialect gives the training-time candidate
    rows (argument hints included), never a KeyError or silently empty argument hints."""
    head, _, dps = rich
    adapter = OpenAIMessagesAdapter(tools=openai_tools())  # the training rows: candidates from the catalog
    feats = ExtractorRunner(CODE, trust_extractor=True).run([dps[0].prefix])
    row = rows_to_table(head.card, [dps[0]], feats, adapter)
    training = dict(zip(row.df[CANDIDATE], head.agent.score(row), strict=True))
    assert any(row.df["candidate_args"])  # the catalog's parameter names
    for tools in (
        openai_tools(),
        messages_api_tools(),
        _mcp_tools(),
        [t["function"] for t in openai_tools()],
    ):
        out = head.decide(dps[0].prefix, tools)
        assert out["scores"] == pytest.approx(training), tools[0]
    assert head.decide(dps[0].prefix, list(parse_catalog(openai_tools()).values()))["scores"] == out["scores"]
    with pytest.raises(ValueError, match="candidate"):
        head.decide(dps[0].prefix, [{"description": "a tool without a name"}])


def test_http_service_takes_tool_schemas_and_rejects_malformed_candidates(rich):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tabagent.serve import create_app

    head, _, dps = rich
    client = TestClient(create_app(head))
    r = client.post("/decide", json={"prefix": dps[0].prefix, "candidates": openai_tools(), "k": 2})
    assert r.status_code == 200 and r.json()["scores"] == pytest.approx(
        head.decide(dps[0].prefix, openai_tools())["scores"]
    )
    r = client.post("/decide", json={"prefix": dps[0].prefix, "candidates": [{"description": "nameless"}]})
    assert r.status_code == 400 and "candidate" in r.json()["detail"]


def test_invalid_unicode_in_a_live_run_is_scored_not_a_crash(rich):
    """A lone surrogate (UTF-16 text cut inside a pair, which JSON carries as an escape) reaches the text
    encoder as '?', in every serving path, instead of crashing the decision."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tabagent.serve import create_app

    head, _, dps = rich
    prefix = json.loads(json.dumps(dps[0].prefix))
    prefix["steps"][-1]["content"] += " ✈️ 予約 \ud83d"
    clean = json.loads(json.dumps(prefix).replace("\\ud83d", "?"))
    out = head.decide(prefix, openai_tools(), k=2)
    assert out["error"] is None and out["scores"] == pytest.approx(
        head.decide(clean, openai_tools(), k=2)["scores"]
    )
    body = json.dumps(
        {"prefix": prefix, "candidates": openai_tools(), "k": 2}
    )  # ASCII-escaped, as JSON allows
    r = TestClient(create_app(head)).post(
        "/decide", content=body, headers={"content-type": "application/json"}
    )
    assert r.status_code == 200 and r.json()["candidates"] == out["candidates"]


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_warm_workers_cold_sandbox_and_in_process_give_identical_decisions(trained):
    """The warm pool (the serving default) answers exactly as the cold sandbox and the trusted in-process
    mode, for decisions that succeed and for one whose extractor fails."""
    _, _, _, dps = trained
    names = [t["function"]["name"] for t in openai_tools()]
    prefixes = [dp.prefix for dp in dps[:6]] + [{"steps": []}]  # the last lacks 'instruction'
    heads = [
        _head(trained, trust_extractor=True),
        _head(trained, trust_extractor=False),
        _head(trained, trust_extractor=False, sandbox="subprocess"),
    ]
    answers = [[h.decide(p, names, k=2) for p in prefixes] for h in heads]
    assert answers[0] == answers[1] == answers[2]
    assert answers[1][-1]["deferred"] and "'instruction'" in answers[1][-1]["error"]
    pool = heads[1].extractor_pool
    assert isinstance(pool, WarmSandbox) and pool.stats()["started"] == 1  # one worker served all 7
    assert heads[0].extractor_pool is None and heads[2].extractor_pool is None
    heads[1].close()
    assert pool.stats()["live"] == 0


def test_a_head_warms_its_pool_and_traces_its_decisions(trained):
    _, _, _, dps = trained
    head = _head(trained, trust_extractor=False, sandbox="warm:2")
    assert head.warm() == 2 and head.extractor_pool.ready
    trace = {}
    out = head.decide(dps[0].prefix, ["search_flights", "book_flight", "get_weather"], k=1, trace=trace)
    assert out["error"] is None and trace["error_kind"] is None
    assert set(trace) == {"prefix_s", "extraction_s", "scoring_s", "error_kind", "frame"}
    assert list(trace["frame"].columns) == head.card.feature_names and len(trace["frame"]) == 3
    head.close()
    assert _head(trained).warm() == 0  # in-process: nothing to start


@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
@pytest.mark.parametrize("spec", [None, "warm:1", "subprocess"])
def test_an_extractor_that_breaks_the_extraction_rules_is_never_run(trained, spec):
    """A sandbox built from a name re-applies the compile step's security checks to extract.py: code that
    imports os is refused, every decision is deferred, and warm() says why."""
    from tabagent.tabschema.warm import ExtractorBrokenError

    agent, adapter, _, dps = trained
    evil = "import os\nfeatures = {'goal': os.getcwd(), 'first_user': ''}\n"
    head = TabAgentHead(agent, TOOL_SHORTLIST, evil, adapter=adapter, sandbox=spec)
    trace = {}
    out = head.decide(dps[0].prefix, ["search_flights", "book_flight"], k=1, trace=trace)
    assert out["deferred"] and "violates the extraction rules" in out["error"]
    assert "import: os" in out["error"]
    assert trace["error_kind"] == "broken"
    with pytest.raises(ExtractorBrokenError, match="import: os"):
        head.warm()
    head.close()


def test_malformed_requests_raise_request_errors(trained):
    from tabagent.api import RequestError

    head = _head(trained)
    _, _, _, dps = trained
    for kwargs in (
        {"candidates": None},
        {"candidates": [{"description": "a tool without a name"}]},
        {"candidates": ["a"], "run": {"messages": []}},
    ):
        with pytest.raises(RequestError):
            head.decide(dps[0].prefix, **kwargs)
    with pytest.raises(RequestError, match="cannot build the prefix"):
        head.decide(run="not a run", candidates=["a"])
