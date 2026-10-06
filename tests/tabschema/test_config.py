"""Every ``TabSchemaConfig`` field is checked when the config is made, with a message naming the field; the
CLI exposes the knobs that matter (with a ``--config`` file the flags override)."""

from __future__ import annotations

import json

import pytest
import yaml

from tabagent import cli
from tabagent.tabschema import TabSchemaConfig

BAD = [
    ("n_analyze", 0),
    ("n_analyze", 2.5),
    ("n_analyze", True),
    ("sample_seed", "7"),
    ("sample_strategy", "random"),
    ("max_code_retries", -1),
    ("json_retries", None),
    ("accept", "accept"),
    ("accept", ("accepted",)),
    ("accept", ()),
    ("compile_min_success", 0),
    ("compile_min_success", 1.5),
    ("max_null_rate", -0.1),
    ("max_columns_per_feature", 0),
    ("max_extractor_columns", "512"),
    ("compile_repair", "yes"),
    ("drop_redundant_text", None),
    ("dependency_features", ("co_occurrence",)),
    ("pair_embeddings", "auto"),
    ("workers", 0),
    ("timeout_s", 0),
    ("timeout_s", float("inf")),
    ("max_cost_usd", -5),
    ("max_cost_usd", "5"),
    ("prices", {"m": (1.0,)}),
    ("prices", {"m": (1.0, -2.0)}),
    ("prices", [("m", 1, 2)]),
    ("max_prompt_chars", 10),
    ("stage_models", {"coder": "x"}),
    ("stage_models", {"extractor": ""}),
    ("fallback_model", ""),
    ("model", 5),
    ("context_text", 1),
    ("verify", None),
    ("verify_adapter", "no"),
    ("verify_samples", 0),
    ("goldens", -1),
    ("audit", "off"),
    ("audit_samples", 0),
    ("audit_max_incorrect", 1.0),
    ("usefulness", 0),
    ("prune_harmful", "x"),
    ("usefulness_max_decisions", 1),
    ("usefulness_max_rows", 0),
    ("llm_retries", -1),
    ("request_timeout_s", 0),
    ("request_timeout_s", "60"),
]


@pytest.mark.parametrize("field, value", BAD, ids=[f"{f}={v!r}" for f, v in BAD])
def test_every_field_is_checked(field, value):
    with pytest.raises(ValueError, match=field):
        TabSchemaConfig(**{field: value})


def test_every_field_has_a_check():
    checked = {f for f, _ in BAD}
    fields = set(TabSchemaConfig.__dataclass_fields__)
    assert fields - checked == set()  # a new field needs its check and a case here


def test_good_values_are_normalized():
    cfg = TabSchemaConfig(
        accept=["accept"], dependency_features=["co_usage", "co_usage"], prices={"m": [1, 2]}
    )
    assert cfg.accept == ("accept",) and cfg.dependency_features == ("co_usage",)
    assert cfg.prices == {"m": (1.0, 2.0)}


def test_from_dict_names_unknown_fields_and_round_trips():
    with pytest.raises(ValueError, match="'n_analyse' \\(did you mean 'n_analyze'\\?\\)"):
        TabSchemaConfig.from_dict({"n_analyse": 5})
    with pytest.raises(ValueError, match="mapping"):
        TabSchemaConfig.from_dict(["n_analyze", 5])
    cfg = TabSchemaConfig(n_analyze=7, accept=("accept",), prices={"m": (1.0, 2.0)})
    again = TabSchemaConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg


def args_for(*flags: str):
    return cli.build_parser().parse_args(["onboard", "logs", *flags])


def test_the_cli_maps_its_flags_onto_the_config(tmp_path):
    cfg = cli.tabschema_config(
        args_for(
            "--n-analyze",
            "7",
            "--accept",
            "accept",
            "--sample-strategy",
            "tasks",
            "--workers",
            "3",
            "--timeout",
            "12",
            "--llm-retries",
            "2",
            "--audit-samples",
            "5",
            "--max-cost",
            "9",
            "--price-in",
            "1",
            "--price-out",
            "4",
            "--stage-model",
            "extractor=other",
            "--price",
            "other=2,8",
            "--no-usefulness",
            "--no-compile-repair",
            "--no-context-text",
            "--no-verify-adapter",
            "--prune-harmful",
        ),
        "main-model",
    )
    assert (cfg.n_analyze, cfg.accept, cfg.sample_strategy, cfg.workers) == (7, ("accept",), "tasks", 3)
    assert (cfg.timeout_s, cfg.llm_retries, cfg.audit_samples, cfg.max_cost_usd) == (12.0, 2, 5, 9.0)
    assert cfg.prices == {"main-model": (1.0, 4.0), "other": (2.0, 8.0)}
    assert cfg.stage_models == {"extractor": "other"}
    assert not (cfg.usefulness or cfg.compile_repair or cfg.context_text or cfg.verify_adapter)
    assert cfg.verify and cfg.audit and cfg.prune_harmful
    default = cli.tabschema_config(args_for(), "m")
    assert default == TabSchemaConfig()


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_a_config_file_is_read_and_the_flags_override_it(tmp_path, fmt):
    fields = {"n_analyze": 10, "audit_samples": 6, "accept": ["accept"], "stage_models": {"judges": "j"}}
    path = tmp_path / f"cfg.{fmt}"
    path.write_text(json.dumps(fields) if fmt == "json" else yaml.safe_dump(fields))
    cfg = cli.tabschema_config(args_for("--config", str(path), "--n-analyze", "4"), "m")
    assert (cfg.n_analyze, cfg.audit_samples, cfg.accept) == (4, 6, ("accept",))
    assert cfg.stage_models == {"judges": "j"}


def test_config_mistakes_are_one_line_user_errors(tmp_path, capsys):
    bad = tmp_path / "cfg.json"
    bad.write_text(json.dumps({"n_analyse": 3}))
    assert cli.main(["onboard", str(tmp_path), "--config", str(bad), "--dry-run"]) == 2
    assert "did you mean 'n_analyze'" in capsys.readouterr().err
    assert cli.main(["onboard", str(tmp_path), "--accept", "accept,maybe", "--dry-run"]) == 2
    assert "TabSchemaConfig.accept: unknown value(s) ['maybe']" in capsys.readouterr().err
    assert cli.main(["onboard", str(tmp_path), "--n-analyze", "0", "--dry-run"]) == 2
    assert "n_analyze must be an integer of at least 1" in capsys.readouterr().err
