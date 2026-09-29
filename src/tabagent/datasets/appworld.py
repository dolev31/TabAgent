"""AppWorld sub-agent routing (paper Table 1, policy family; App. "AppWorld").

AppWorld (trivedi2024appworld) publishes the experiment outputs of its baseline agent systems: four agent
architectures (ReAct, plan-and-execute, full-code, function calling) over four LLM backbones, 14 systems in
all, each run on all 585 test tasks (168 test_normal and 417 test_challenge) and scored by the official
evaluator. This executed matrix is dense, so choosing a system per task is a policy decision. Each row is
one (task, system) pair:

  task_uid            group id (the trajectory-corpus id: a hash of the AppWorld task id)
  task_id             AppWorld task id; scenario_id is its scenario (tasks ``<scenario>_1..3``)
  official_split      test_normal or test_challenge
  instruction         the task instruction, the only pre-decision context
  system_id           the candidate system, ``<agent>/<provider>/<model>``
  success             all evaluator tests passed (label and reward)
  eval_pass_fraction  fraction of the evaluator tests passed, kept for scoring only

**Obtaining the data.** AppWorld's data and experiment outputs are Apache-2.0 with an added condition: any
public redistribution must stay encrypted. TabAgent therefore never downloads or ships them. Install
AppWorld and fetch both with its official CLI (``pip install appworld``, ``appworld install``,
``appworld download data``, ``appworld download experiment-outputs``), then set ``APPWORLD_ROOT`` to that
root (or pass ``root=``). The builder reads, under the root:

  data/version.txt                                   must be 0.2.0
  data/datasets/{test_normal,test_challenge}.txt     the test task ids
  data/tasks/<task_id>/specs.json                    the instruction
  experiments/outputs/<system>/<split>/tasks/<task_id>/evaluation/report.md
                                                     the official evaluator report of each baseline run

The built table contains plaintext task instructions. It stays in the local cache; do not publish it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from tabagent.datasets._core import DatasetInfo, Source
from tabagent.evaluation.cv import stable_uid

DATA_VERSION = "0.2.0"
OUTPUTS_VERSION = "0.1.3"

SOURCE = Source(
    id="appworld_official",
    benchmark="AppWorld",
    url="https://github.com/StonyBrookNLP/appworld",
    revision="a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
    release_id=f"data-{DATA_VERSION}_outputs-{OUTPUTS_VERSION}",
    license="Apache-2.0, with the added condition that data and outputs are redistributed only encrypted",
    redistribution="access_restricted",
    citation="trivedi2024appworld",
    notes=(
        f"Data {DATA_VERSION} and baseline experiment outputs {OUTPUTS_VERSION} are obtained only through "
        "AppWorld's official CLI; TabAgent reads them from APPWORLD_ROOT and never redistributes them."
    ),
)

SPLITS = ("test_normal", "test_challenge")
SYSTEMS = tuple(
    f"legacy_{agent}_agent/{backbone}"
    for agent in ("full_code", "function_calling", "plan_execute_code", "react_code")
    for backbone in (
        "deepseek/deepseek-coder-33b-instruct_together",
        "meta/llama-3-70b-chat-hf_together",
        "openai/gpt-4-turbo-2024-04-09",
        "openai/gpt-4o-2024-05-13",
    )
    if agent != "function_calling" or backbone.startswith("openai/")
)
BACKBONES = ("gpt-4o", "gpt-4-turbo", "llama-3-70b", "deepseek-coder-33b")

SETUP = f"""AppWorld is not redistributed with TabAgent (its license requires encrypted redistribution).
Obtain the data and the baseline experiment outputs with AppWorld's official CLI, then point TabAgent at
that directory:

    pip install appworld
    appworld install
    appworld download data --version {DATA_VERSION} --root <dir>
    appworld download experiment-outputs --version {OUTPUTS_VERSION} --root <dir>
    export APPWORLD_ROOT=<dir>        # or pass root=<dir> to the builder
"""


# --------------------------------------------------------------------------- stateless card features
def _architecture(system: str) -> str:
    return system.split("/")[0].replace("legacy_", "").replace("_agent", "")


def _backbone(system: str) -> str:
    model = system.split("/", 1)[1]
    return next((b for b in BACKBONES if b in model), model)


def _backbone_family(backbone: str) -> str:
    return "openai" if backbone.startswith("gpt") else "meta" if backbone.startswith("llama") else "deepseek"


def architecture(df: pd.DataFrame, col: str = "system_id") -> pd.Series:
    return df[col].map(_architecture)


def backbone(df: pd.DataFrame, col: str = "system_id") -> pd.Series:
    return df[col].map(_backbone)


def backbone_family(df: pd.DataFrame, col: str = "system_id") -> pd.Series:
    return df[col].map(lambda s: _backbone_family(_backbone(s)))


# --------------------------------------------------------------------------- source files
_STAT = {"passed": "Num Passed Tests", "failed": "Num Failed Tests", "total": "Num Total  Tests"}


def parse_report(path: Path) -> dict[str, int]:
    """Test counts of one evaluator ``report.md``; success means no failed test (and at least one test)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, label in _STAT.items():
        m = re.search(rf"{re.escape(label)}\s*:\s*(\d+)", text)
        if m is None:
            raise ValueError(f"cannot parse the AppWorld evaluation report {path}")
        out[key] = int(m.group(1))
    return out


def appworld_root(root: str | Path | None = None) -> Path:
    """The AppWorld root (``root`` or ``$APPWORLD_ROOT``), checked for the data and experiment outputs."""
    root = root or os.environ.get("APPWORLD_ROOT")
    if not root:
        raise FileNotFoundError("APPWORLD_ROOT is not set.\n" + SETUP)
    root = Path(root).expanduser()
    for sub in ("data/datasets", "experiments/outputs"):
        if not (root / sub).is_dir():
            raise FileNotFoundError(f"{root / sub} does not exist.\n" + SETUP)
    vfile = root / "data" / "version.txt"
    version = vfile.read_text().strip() if vfile.is_file() else "of unknown version"
    if version != DATA_VERSION:
        raise ValueError(f"AppWorld data {version} found, the paper uses {DATA_VERSION}.\n" + SETUP)
    return root


# --------------------------------------------------------------------------- builder
def build_subagent_routing(
    source: Source = SOURCE, download: bool = False, root: str | Path | None = None
) -> pd.DataFrame:
    """The dense (test task x system) table. ``download`` is ignored: AppWorld comes only from its CLI."""
    root = appworld_root(root)
    data, outputs = root / "data", root / "experiments" / "outputs"
    rows, missing = [], []
    for split in SPLITS:
        ids = (data / "datasets" / f"{split}.txt").read_text(encoding="utf-8").split()
        for task_id in sorted(ids):
            specs = json.loads((data / "tasks" / task_id / "specs.json").read_text(encoding="utf-8"))
            base = {
                "task_uid": stable_uid("task", SOURCE.id, SOURCE.release_id, task_id),
                "task_id": task_id,
                "scenario_id": task_id.rsplit("_", 1)[0],
                "official_split": split,
                "instruction": specs.get("instruction") or "",
            }
            for system in SYSTEMS:
                report = outputs / system / split / "tasks" / task_id / "evaluation" / "report.md"
                if not report.is_file():
                    missing.append(report)
                    continue
                r = parse_report(report)
                rows.append(
                    {
                        **base,
                        "system_id": system,
                        "success": int(r["failed"] == 0 and r["total"] > 0),
                        "eval_pass_fraction": r["passed"] / r["total"] if r["total"] else 0.0,
                    }
                )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} baseline evaluation reports are missing (first: {missing[0]}). Download the "
            f"experiment outputs {OUTPUTS_VERSION}.\n" + SETUP
        )
    if not rows:
        raise ValueError(f"no AppWorld test tasks listed under {data / 'datasets'}")
    return pd.DataFrame(rows).sort_values(["task_id", "system_id"]).reset_index(drop=True)


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="appworld/subagent-routing",
        source=SOURCE,
        build=build_subagent_routing,
        card="appworld_subagent_routing",
        folds="appworld_subagent_routing",
        decision="sub-agent routing (policy): route each AppWorld test task to one of 14 agent systems",
        paper="Table 1, sub-agent routing: route@1 38 (best fixed system 36, chance 16, oracle 58)",
        description=(
            "585 AppWorld test tasks x 14 official baseline systems (8,190 rows); label and reward = the "
            "run passed every evaluator test. Needs APPWORLD_ROOT (see tabagent.datasets.appworld)."
        ),
    )
]
