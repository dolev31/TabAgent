"""Text columns that duplicate another kept text column word for word (the same lowercased tokens, numbers and
order on every decision point) are dropped at compile, keeping one; any difference keeps both."""

from __future__ import annotations

import json

from test_robustness import FakeLLM, discover, spec, tagged

from tabagent.tabschema.compile import CompiledExtractor, _drop_duplicate_text, _sequence

FORMS = (
    "intent = trajectory['intent']\n"
    "words = intent.lower().split()\n"
    "counts = {}\n"
    "for w in words:\n"
    "    counts[w] = counts[w] + 1 if w in counts else 1\n"
    "plan = [s['data'] for s in trajectory['steps'] if s['name'] == 'PlanControllerAgent'][-1]\n"
    "features = {\n"
    "    'query_text': intent[0].upper() + intent[1:] + '.',\n"
    "    'query_tokens': json.dumps(words),\n"
    "    'query_bag_of_words': json.dumps(counts, sort_keys=True),\n"
    "    'query_unigrams_and_bigrams': ' '.join(words + [a + '_' + b for a, b in zip(words, words[1:])]),\n"
    "    'plan_text': plan,\n"
    "}\n"
)
NAMES = [
    ("query_text", "text"),
    ("query_tokens", "text"),
    ("query_bag_of_words", "text"),
    ("query_unigrams_and_bigrams", "text"),
    ("plan_text", "text"),
]


def fake() -> FakeLLM:
    proposals = [spec(n, t) for n, t in NAMES]
    return FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": proposals}),
        extractor=lambda p, n: tagged("import json\n" + FORMS),
        aggregator=lambda p, n: json.dumps({"filtered_features": proposals}),
    )


def test_a_text_and_its_tokens_collapse_and_other_forms_stay(tmp_path):
    """The four forms of a real shortlister card: raw text and its token list are the same sequence; the bag
    of words (with counts, sorted) and the n-grams are other content and stay."""
    result = discover(tmp_path, fake())
    ext = result.extractor
    assert ext.dropped["query_tokens"].startswith("redundant with query_text")
    assert {"query_text", "query_bag_of_words", "query_unigrams_and_bigrams", "plan_text"} <= set(
        ext.features
    )
    record = json.loads((tmp_path / "runs" / "r1" / "compile.json").read_text())
    assert record["dropped"]["query_tokens"].startswith("redundant with query_text")
    drops = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["drops_by_reason"]
    assert drops["redundant text (the same words as another kept text column)"] == 1
    off = discover(tmp_path, fake(), run_id="off", drop_redundant_text=False)
    assert not any(r.startswith("redundant with") for r in off.extractor.dropped.values())


def compiled(columns: dict[str, list], types: dict[str, str] | None = None) -> CompiledExtractor:
    n = len(next(iter(columns.values())))
    feats = {c: (types or {}).get(c, "text") for c in columns}
    out = CompiledExtractor("", feats, specs={c: {"feature_name": c} for c in columns})
    out.outputs = [{c: v[i] for c, v in columns.items()} for i in range(n)]
    return out


def run(columns: dict[str, list], types=None, context=None) -> dict[str, str]:
    c = compiled(columns, types)
    if context is not None:
        c.features[context[0]] = "text"
        c.specs[context[0]] = {"feature_name": context[0], "added_by": "tabagent"}
    _drop_duplicate_text(c, context)
    return c.dropped


def test_numbers_order_and_negation_keep_both():
    n = 200
    assert not run(
        {
            "requested_flight": [f"flight UA {100 + i}" for i in range(n)],
            "booked_flight": [f"flight UA {900 - (i % 7)}" for i in range(n)],
        }
    )
    assert not run(
        {
            "tools_called_in_order": [
                "search then book then pay" if i % 2 else "pay then book then search" for i in range(n)
            ],
            "tools_planned": ["search then pay then book" for _ in range(n)],
        }
    )
    assert not run(
        {
            "policy_statement": ["the fare is not refundable after departure"] * n,
            "user_belief": ["the fare is refundable after departure"] * n,
        }
    )


def test_a_column_that_differs_on_most_rows_is_kept_whatever_a_sample_sees():
    """20,000 decision points, 10 per run in run order: a strided sample would see only each run's first
    decision, where the history equals the query. Every row is compared, so the history is kept."""
    query, history = [], []
    for r in range(2000):
        q = f"please book a flight to city{r} for my family"
        for k in range(10):
            query.append(q)
            history.append(q + "".join(f" then step{j} tool{j} returned result{j} ok" for j in range(k)))
    assert not run({"query": query, "turn_history": history})
    one_row = [f"go to city{i}" for i in range(5000)]
    other = list(one_row)
    other[4321] += " now"  # a single row of difference keeps both
    assert not run({"a": one_row, "b": other})


def test_a_duplicate_of_the_context_text_is_dropped_and_the_context_text_kept():
    q = [f"book a flight to city{i} tomorrow" for i in range(200)]
    dropped = run({"user_request": [x.upper() + "!" for x in q]}, context=("context_text", q))
    assert dropped["user_request"].startswith("redundant with context_text")


def test_a_category_is_kept_over_its_text_and_ties_break_by_name():
    labels = ["Book Flight", "Cancel Order", "Weather"] * 50
    dropped = run(
        {"intent_text": labels, "intent_label": [x.lower() for x in labels]}, {"intent_label": "cat"}
    )
    assert set(dropped) == {"intent_text"} and dropped["intent_text"].startswith(
        "redundant with intent_label"
    )
    dropped = run({"b": ["go home", "stay"] * 10, "a": ["Go home.", "STAY"] * 10})
    assert set(dropped) == {"b"}  # same type: the first name is kept


def test_sequences_keep_digits_and_order_and_ignore_punctuation():
    assert _sequence("Find my Venmo friends.") == _sequence('["find","my","venmo","friends"]')
    assert _sequence("flight 101") != _sequence("flight 901")
    assert _sequence("a b") != _sequence("b a")
    assert _sequence('{"find": 1}') == ("find", "1")
    assert _sequence(None) == () and _sequence(float("nan")) == () and _sequence(True) == ()
