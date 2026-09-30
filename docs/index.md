# TabAgent

**Replace an LLM agent's repeated closed-set decisions with a cheap classifier trained on the agent's own
successful runs.** Code for the paper *TabAgent: Distilling LLM Agent Components into Cheap Classifiers*.

<!-- named -->
Ido Levy, Segev Shlomov, Avi Yaeli, Nir Mashkif, Asaf Adi, Yinon Goldshtein, Eilam Shapira, Leshem Choshen

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/dolev31/TabAgent/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/dolev31/TabAgent/blob/main/pyproject.toml)
[![CI](https://github.com/dolev31/TabAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/dolev31/TabAgent/actions/workflows/ci.yml)
<!-- /named -->

Agents call a frontier LLM again and again for decisions whose answer comes from a fixed set: which tools to
shortlist, which skill or application to use, which model or sub-agent to route to, whether to activate a
skill. TabAgent learns those decisions from the agent's execution traces and serves them from a tabular
classifier, with the surrounding agent left unchanged.

| AppWorld tool shortlisting, per decision | GPT-5.4 shortlister | TabAgent head |
|---|---|---|
| latency | 7.50 s | 2.682 ms (more than 10³× lower) |
| cost | $0.052 | $2.0×10⁻⁷ (about 10⁵× lower) |
| quality | the reference | recovers essentially all of the tools the shortlister used on its successful runs (Recall@9) |

<p align="center">
  <img src="assets/pareto.png" width="640"
       alt="Per-decision latency against shortlisting quality: TabAgent is the fastest and cheapest point, near the top on quality">
</p>

*Per-decision latency (x, log ms) against shortlisting quality P@R (y). TabAgent (star) is the fastest and
cheapest point while remaining near the top on quality; filled circles are fine-tuned small LLMs, open circles
the same models few-shot. Figure from the paper.*

What the paper finds, across three decision families and seven agent benchmarks:

- **Set retrieval.** On AppWorld tool shortlisting, TabAgent recovers essentially all of the tools the GPT-5.4
  shortlister used on its successful runs (Recall@9) at far lower latency and cost.
- **Classification and policy.** It ranks first on the classification decisions and is on par with a strong
  fixed default on the harder routing and gating decisions, where a costly generative router does worse.
- **Representation, not model.** A discriminative head over raw text or embeddings alone stays near-random;
  the trace-derived features make it accurate.
- **In the loop.** Placed on a live agent's critical path, TabAgent drives AppWorld tasks to completion about
  as often as the GPT-5.4 shortlister it replaces.
- **Few traces.** The head reaches 90% of its full-pool P@R from 118 successful trajectories on AppWorld and
  from 37 to 51 on the four τ²-bench agents.

## How it works

```mermaid
flowchart LR
    T[successful agent<br/>trajectories] --> S[TabSchema<br/>LLM-driven feature discovery]
    S --> C[feature card<br/>+ compiled extractor]
    C --> Y[TabSynth<br/>label-preserving paraphrases]
    Y --> H[TabHead<br/>CatBoost over the card]
    H --> D[decision:<br/>top-k / label / route]
```

1. **TabSchema** reads your agent's own logs and, with an LLM, proposes behavior-grounded features, writes and
   sandbox-tests the code that extracts them, and keeps only those a judge ensemble accepts. The output is a
   *feature card* plus a reusable, reviewed extractor. It runs once per agent and catalog.
2. **TabSynth** adds deterministic, label-preserving paraphrases of the free-text state where trajectories are
   scarce.
3. **TabHead** scores every candidate of a decision in one pass (CatBoost), with an optional confidence gate that
   defers uncertain decisions back to the original LLM.

<p align="center">
  <img src="assets/tabschema.png" width="760" alt="TabSchema: feature analyzer, code extractor and validator, feature aggregator, three judges and a meta-judge produce the feature schema">
  <br><em>TabSchema compiles trajectory-derived schema, state, and dependency signals into tabular features.</em>
</p>
<p align="center">
  <img src="assets/tabhead.png" width="760" alt="TabHead: numerical, categorical and textual features plus the tool catalog go into one tabular-textual unit that outputs a ranked shortlist">
  <br><em>TabHead consumes features and candidates to output per-candidate probabilities.</em>
</p>

## Install

```bash
pip install tabagent                    # core: tables, TabSynth, TabHead, evaluation
# until the PyPI release: pip install "tabagent @ git+https://github.com/dolev31/TabAgent"
pip install "tabagent[tabschema]"       # + onboarding from agent logs (OpenAI-compatible LLM client)
pip install "tabagent[embed]"           # + e5 context-candidate similarity features
pip install "tabagent[serve]"           # + HTTP decision service
```

`pip` resolves the newest dependency releases. For the exact versions pinned in `uv.lock`, clone the
repository and run `uv sync --extra embed`.

## Try it

```bash
git clone https://github.com/dolev31/TabAgent && cd TabAgent
pip install -e ".[embed]"
python examples/01_quickstart_skillsbench.py
```

The first run downloads the pinned SkillsBench release (3.5 MB, SHA-256 verified) and the e5-small-v2 encoder
from Hugging Face (about 128 MB), then runs the paper's five-fold task-level cross-validation on CPU (about
ten minutes on a laptop). It prints P@R ≈ 0.80 and Recall@9 ≈ 0.90 with 95% bootstrap intervals, then the
five skills TabAgent would hand the agent for a held-out task. The paper reports P@R 80 and Recall@9 91;
[docs/reproducing.md](reproducing.md) compares every reproducible cell with the paper's.

| Example | What it shows | Needs |
|---|---|---|
| [01_quickstart_skillsbench.py](https://github.com/dolev31/TabAgent/blob/main/examples/01_quickstart_skillsbench.py) | the paper's components on a public benchmark, cross-validated | CPU; network on first run (3.5 MB of data, 128 MB encoder) |
| [02_bring_your_own_table.py](https://github.com/dolev31/TabAgent/blob/main/examples/02_bring_your_own_table.py) | scikit-learn estimators on your own decision table | CPU, offline |
| [03_onboard_agent_logs.py](https://github.com/dolev31/TabAgent/blob/main/examples/03_onboard_agent_logs.py) | onboarding a decision from your own agent logs | an LLM API key |
| [04_deploy.py](https://github.com/dolev31/TabAgent/blob/main/examples/04_deploy.py) | live decisions in-process, the HTTP service, a tool filter in front of your LLM | a saved head (from 02 or 03) |
| [05_tau2_cross_agent_transfer.py](https://github.com/dolev31/TabAgent/blob/main/examples/05_tau2_cross_agent_transfer.py) | training on one agent's traces and deploying on another's (paper Table 2) | CPU; network on first run (about 600 MB) |

## Three ways to use it

### 1. You have agent logs and an LLM key

```bash
export OPENAI_API_KEY=...        # or OPENAI_BASE_URL for any OpenAI-compatible server, or ANTHROPIC/GEMINI keys
tabagent components ./agent_logs                              # which decisions can be replaced?
tabagent onboard ./agent_logs --decision ToolShortlist --dry-run   # what it will process and cost
tabagent onboard ./agent_logs --decision ToolShortlist --out ./my_head
tabagent serve ./my_head --port 8077
```

```python
import tabagent

head = tabagent.onboard("./agent_logs", decision="ToolShortlist")  # a budget: max_cost_usd= with prices=
print(head.report)  # grouped cross-validation vs BM25 and chance, latency, cost per decision
head.save("my_head")
head.decide(prefix, candidates, k=5)  # {"candidates", "scores", "confidence", "deferred"}
```

Supported log formats are detected automatically: OpenAI / Anthropic / LangChain tool-call message logs,
τ²-bench results, CUGA trajectories, and generic JSON with a `steps` list. Trajectories are sent to your LLM
provider during onboarding; pass `redact=` to scrub them first.

### 2. You have a table of decisions (scikit-learn API)

```python
from sklearn.model_selection import GroupKFold, cross_val_score
import tabagent
from tabagent import TabAgentRanker

# one row per (decision, candidate); y = 1 if the agent chose the candidate
ranker = TabAgentRanker(group_col="task_id", candidate_col="tool")
ranker.fit(X, y)  # infers the card, picks TabSynth and the prior by nested validation
ranker.rank(X_new, k=5)  # top-5 candidates per decision
cross_val_score(ranker, X, y, groups=X.task_id, cv=GroupKFold(5), scoring=tabagent.scorer("p@r"))
print(ranker.report(X, y))
```

`TabAgentClassifier` handles single-label decisions, `TabAgentRouter` handles routing (pick the arm with the
highest predicted success). `tabagent.pointwise()` turns one-row-per-decision logs into this layout.

### 3. Full control: the paper's components

```python
from tabagent import FeatureCard, TabAgent, TabHead, TabSynth, cross_validate, datasets

table = datasets.load("skillsbench/skill-select", download=True)  # rebuilt from the pinned public release
card = datasets.card("skillsbench/skill-select")
agent = TabAgent(card, head=TabHead("ranking"), synth=TabSynth(k=2))
res = cross_validate(agent, table, folds=datasets.folds("skillsbench/skill-select"))
print(res.summary(["p@r", "recall@9"]))
```

## Reproducing the paper

The release reproduces the following from public data: τ²-bench (cross-backbone transfer and per-domain
shortlisting), SkillsBench (skill selection and gating), Mind2Web (operation and element selection),
LLMRouterBench (model routing; MIT as declared upstream, opt-in download) and AppWorld sub-agent routing (data obtained
through AppWorld's official CLI). See [docs/reproducing.md](reproducing.md) for each paper cell, the command
that recomputes it, and the value it produces. The AppWorld tool-shortlisting results and the in-loop hot swap
use the CUGA agent's own trajectories, which are not released; the CUGA trajectory adapter is included, so the
pipeline can be rerun on your own CUGA traces. The OAK and BPO benchmarks are public; loaders for them are not
included yet.

## Documentation

| Guide | Covers |
|---|---|
| [Concepts](concepts.md) | the decision table, the four decision kinds, leakage safety, the evaluation protocol |
| [Components](components.md) | TabSchema, TabSynth and TabHead in detail |
| [Onboard an agent](onboarding.md) | log formats, privacy and cost controls, reviewing the generated extractor |
| [Bring your own table](bring_your_own_data.md) | the scikit-learn estimators and automatic recipe selection |
| [Deploy](deployment.md) | saved heads, the HTTP service, the tool filter, the confidence gate |
| [Datasets](datasets.md) | every public dataset, its pinned source, license and the paper result it backs |
| [Reproducing the paper](reproducing.md) | each paper cell, the command that recomputes it, paper vs released values |

`make docs` builds these pages as a searchable site (mkdocs-material).

## Data policy

TabAgent ships no benchmark data and no agent trajectories. Public datasets are downloaded from their upstream
releases at pinned revisions and verified by SHA-256; licenses are listed in [NOTICE](https://github.com/dolev31/TabAgent/blob/main/NOTICE).

<!-- named -->
## Citation

If you use TabAgent, please cite the paper:

```bibtex
@misc{levy2026tabagent,
  title  = {TabAgent: Distilling LLM Agent Components into Cheap Classifiers},
  author = {Levy, Ido and Shlomov, Segev and Yaeli, Avi and Mashkif, Nir and Adi, Asaf and
            Goldshtein, Yinon and Shapira, Eilam and Choshen, Leshem},
  year   = {2026}
}
```

<!-- /named -->
## License

Apache-2.0.
