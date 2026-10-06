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


# --------------------------------------------------------------------------- one form per text feature
def test_the_derived_forms_of_one_feature_collapse_into_its_text(tmp_path):
    """Four forms of one text as the columns of ONE planned feature (fabricated values): the token list
    repeats the text word for word; the bag of words restates its words in a shorter derived form. The text
    is kept (the head represents text itself), and so are the n-grams, a longer plain text."""
    proposals = [spec("query", "text"), spec("plan_text", "text")]
    llm = FakeLLM(
        analyzer=lambda p, n: json.dumps({"potential_features": proposals}),
        extractor=lambda p, n: tagged("import json\n" + FORMS),
        aggregator=lambda p, n: json.dumps({"filtered_features": proposals}),
    )
    ext = discover(tmp_path, llm).extractor
    assert ext.dropped["query_tokens"].startswith("redundant with query_text")
    assert (
        ext.dropped["query_bag_of_words"] == "a derived form of query_text; the head represents text itself"
    )
    assert {"query_text", "query_unigrams_and_bigrams", "plan_text"} <= set(ext.features)
    drops = json.loads((tmp_path / "runs" / "r1" / "run.json").read_text())["drops_by_reason"]
    assert drops["derived text form (tokens, bag of words or n-grams of a kept text column)"] == 1


def _derived_run(columns: dict[str, list], feature: dict[str, str] | None = None) -> dict[str, str]:
    from tabagent.tabschema.compile import _drop_derived_text

    c = compiled(columns)
    for col in columns:
        c.specs[col] = {"feature_name": (feature or {}).get(col, "query")}
    _drop_derived_text(c)
    return c.dropped


def test_derived_forms_only_within_one_feature_and_only_on_every_row():
    text = [f"Find my friends in city{i}." for i in range(50)]
    words = [json.dumps(t.lower().strip(".").split()) for t in text]
    bag = [json.dumps({w: 1 for w in t.lower().strip(".").split()}) for t in text]
    grams = [
        " ".join(
            [*(w := t.lower().strip(".").split()), *(a + "_" + b for a, b in zip(w, w[1:], strict=False))]
        )
        for t in text
    ]
    dropped = _derived_run(
        {"query_text": text, "query_bag": bag, "query_ngrams": grams, "query_tokens": words}
    )
    assert set(dropped) == {"query_bag", "query_tokens"}  # the n-grams are longer: kept
    assert all(r.startswith("a derived form of query_text") for r in dropped.values())
    # another feature's forms are never compared (#43 alone handles columns across features)
    assert not _derived_run({"a_text": text, "b_bag": bag}, {"a_text": "a", "b_bag": "b"})
    # a first sentence, a different word on one row, or new words: kept
    first = [t.split(" in ")[0] for t in text]
    assert not _derived_run({"query_text": text, "query_first": first})
    odd = list(bag)
    odd[17] = json.dumps({"elsewhere": 1})
    assert not _derived_run({"query_text": text, "query_bag": odd})
    assert _sequence(None) == ()


def test_the_longer_and_the_ordered_form_are_never_dropped():
    """Whatever the names and the lengths: the plain text over its sorted, de-duplicated words, and an ordered
    sequence over its set."""
    text = [f"find my friends in city{i} and my family" for i in range(50)]
    sorted_words = [" ".join(sorted(set(t.split()))) for t in text]
    for a, b in (("query_text", "query_bag"), ("query_bag", "query_text"), ("a_text", "z_bag")):
        dropped = _derived_run({a: text, b: sorted_words})
        assert set(dropped) == {b} and dropped[b].startswith(f"a derived form of {a}")
    # the same words in another order that is not sorted: both kept
    shuffled = [" ".join(reversed(t.split())) for t in text]
    assert not _derived_run({"query_text": text, "query_reversed": shuffled})
    calls = [["search", "book", f"pay{i}"] for i in range(50)]  # an order of its own, no repeats
    seq = [json.dumps(c) for c in calls]
    as_set = [json.dumps(sorted(set(c))) for c in calls]
    as_dict = [json.dumps(dict.fromkeys(c, 1)) for c in calls]
    as_text = [" ".join(sorted(set(c))) for c in calls]
    for a, b in (("api_sequence", "api_set"), ("api_set", "api_sequence"), ("a_set", "z_sequence")):
        sequence, other = (a, b) if "sequence" in a else (b, a)
        for values in (as_set, as_dict):
            dropped = _derived_run({sequence: seq, other: values})
            assert set(dropped) == {other} and dropped[other].startswith(f"a derived form of {sequence}")
    # an ordered sequence next to text that is only its sorted words: the sequence is kept
    assert "api_sequence" not in _derived_run({"api_sequence": seq, "api_text": as_text})
    # repeats are an order of their own too: kept, and the set of them dropped
    repeats = [json.dumps(["search", "search", f"pay{i}"]) for i in range(50)]
    unique = [json.dumps(sorted({"search", f"pay{i}"})) for i in range(50)]
    assert set(_derived_run({"api_calls": repeats, "api_set": unique})) == {"api_set"}
