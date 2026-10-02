---
description: "TabAgent replaces an LLM agent's repeated closed-set decisions, such as tool shortlisting and routing, with a cheap classifier trained on its own traces."
# For search and answer engines (docs/overrides/main.html renders these into the page's <head>). They mirror the
# Abstract and Key terms sections below, which show the same words: keep the two the same.
keywords: ["LLM agents", "tool shortlisting", "tool selection", "routing", "LLM routing", "skill selection", "tabular classification", "distillation", "agent efficiency"]
abstract: >-
  Agentic systems chain LLM calls to make closed-set decisions: choosing from a fixed candidate set, as in skill
  selection, tool shortlisting, application selection, and routing. These repeated frontier-model calls dominate
  per-task latency and cost. We show such decisions need not be made by a language model at all: a compact
  discriminative head, trained on the agent's own successful execution traces, can replace the generative one while
  leaving the surrounding agent intact. TabAgent has three parts. (i) TabSchema extracts schema, state, and
  dependency features from trajectories. (ii) TabSynth adds schema-aligned synthetic rows where trajectory coverage is
  thin. (iii) TabHead, a cheap tabular classifier, scores candidates in a single pass. Changing only the label space,
  we test TabAgent as a drop-in replacement for the frontier-LLM heads (e.g., GPT-5.4) across three decision families
  and seven agent benchmarks. On tool shortlisting it recovers essentially all of the tools the GPT-5.4
  shortlister used on its successful runs (Recall@9), at 99.96% lower latency and 99.9996% lower cost per call
  (∼10³× and ∼10⁵×). It also ranks first on the classification decisions and is on par with a strong fixed default
  on the harder routing and gating decisions at far lower latency and cost, where a costly generative router does
  worse. Across these tasks, the gains come from the representation, not the model: a discriminative head over raw
  text or embeddings alone collapses, and trace-derived features make it accurate.
terms:
  - name: "Closed-set decision"
    definition: "A decision inside an agent's execution whose answer is selected from a fixed, bounded set of candidates: which skill to invoke, which tool to call next, which application to route to, or which sub-task to gate."
  - name: "Tool shortlisting"
    definition: "The task of picking a small subset from a catalog of hundreds to thousands of tools, where a tool is any externally callable capability (API endpoint, function, or GUI operation) with a typed schema and observable outputs."
  - name: "Feature card"
    definition: "The tabular layout TabAgent learns from, with one row per (context, candidate) pair and one column per behavior-grounded feature."
  - name: "TabSchema"
    definition: "An orchestrated agentic system that proposes features, implements them as executable trace-extraction code, and keeps only those a judge ensemble agrees are decision-relevant."
  - name: "TabSynth"
    definition: "A schema-aligned augmentation module: a deterministic, rule-based paraphraser that rewrites the free-text state of an existing training row while copying the candidate, the label, and every schema, state, and dependency feature verbatim."
  - name: "TabHead"
    definition: "The classifier that scores each (context, candidate) pair over the TabSchema feature card in a single pass. It is CatBoost in every main-paper result."
  - name: "P@R (R-precision)"
    definition: "The share of a task's ground-truth tools that appear in the top R of the ranking, where R is the task's relevant-set size, averaged over tasks. It equals Recall@R and captures completeness."
  - name: "Recall@k"
    definition: "The share of a task's ground-truth tools recovered in the top k of the ranking, averaged over tasks: recovery at a fixed budget."
hide:
  - navigation
  - toc
---

<div class="ta-hero" markdown>

<h1 class="ta-hero__title" id="tabagent" markdown>![TabAgent](assets/brand/wordmark-light.svg#only-light) ![TabAgent](assets/brand/wordmark-dark.svg#only-dark)</h1>

<p class="ta-hero__tagline">Replace an LLM agent's repeated <span class="ta-nowrap">closed-set</span> decisions with a cheap classifier trained on the agent's own successful runs.</p>

<p class="ta-hero__paper" markdown>Code for the paper *TabAgent: Distilling LLM Agent Components into Cheap Classifiers*.</p>

<!-- named -->
<p class="ta-hero__authors">Ido&nbsp;Levy<sup>1,2</sup>&nbsp;· Segev&nbsp;Shlomov<sup>1</sup>&nbsp;· Avi&nbsp;Yaeli<sup>1</sup>&nbsp;· Nir&nbsp;Mashkif<sup>1</sup>&nbsp;· Asaf&nbsp;Adi<sup>1</sup>&nbsp;· Yinon&nbsp;Goldstein<sup>1</sup>&nbsp;· Eilam&nbsp;Shapira<sup>1</sup>&nbsp;· Leshem&nbsp;Choshen<sup>1,2</sup></p>
<p class="ta-hero__authors"><sup>1</sup>IBM&nbsp;· <sup>2</sup>Weizmann&nbsp;Institute&nbsp;of&nbsp;Science</p>
<!-- /named -->

<div class="ta-hero__actions" markdown>

[Get started :material-arrow-right:](bring_your_own_data.md){ .md-button .md-button--primary }
<!-- named -->
[Paper](https://arxiv.org/abs/2602.16429){ .md-button data-goatcounter-click="paper" }
<!-- /named -->
[Onboard an agent](onboarding.md){ .md-button }
[Reproduce the paper](reproducing.md){ .md-button }

</div>

<div class="ta-hero__art" markdown>

![Today an LLM agent sends each closed-set decision, such as shortlisting tools, to a frontier-LLM call. With TabAgent, its successful runs pass through TabSchema, TabSynth and TabHead, and one classifier pass returns the ranked shortlist.](assets/brand/hero-light.svg#only-light)
![Today an LLM agent sends each closed-set decision, such as shortlisting tools, to a frontier-LLM call. With TabAgent, its successful runs pass through TabSchema, TabSynth and TabHead, and one classifier pass returns the ranked shortlist.](assets/brand/hero-dark.svg#only-dark)

</div>

</div>

<div class="ta-stats">
  <div class="ta-stat">
    <div class="ta-stat__label">Latency per decision</div>
    <div class="ta-stat__value">2.682 ms</div>
    <div class="ta-stat__compare">vs 7.50 s per decision for the GPT-5.4 shortlister</div>
  </div>
  <div class="ta-stat">
    <div class="ta-stat__label">Cost per decision</div>
    <div class="ta-stat__value">$2.0×10⁻⁷</div>
    <div class="ta-stat__compare">vs $0.052 per decision for the GPT-5.4 shortlister</div>
  </div>
  <div class="ta-stat">
    <div class="ta-stat__label">Recall@9</div>
    <div class="ta-stat__value">92–100</div>
    <div class="ta-stat__compare">of the tools the GPT-5.4 shortlister used on its successful runs</div>
  </div>
</div>
<p class="ta-stats__caption">AppWorld tool shortlisting: TabAgent's head against the GPT-5.4 shortlister it replaces.</p>

Agents call a frontier LLM again and again for decisions whose answer comes from a fixed set: which tools to
shortlist, which skill or application to use, which model or sub-agent to route to, whether to activate a
skill. TabAgent learns those decisions from the agent's execution traces and serves them from a tabular
classifier, with the surrounding agent left unchanged.
{ .ta-lead }

## Guides

<div class="grid cards ta-guides" markdown>

-   :material-lightbulb-on-outline: [Concepts](concepts.md)

    The decision table, the four decision kinds, leakage safety, the evaluation protocol.

-   :material-puzzle-outline: [Components](components.md)

    TabSchema, TabSynth and TabHead in detail.

-   :material-robot-outline: [Onboard an agent](onboarding.md)

    Log formats, privacy and cost controls, reviewing the generated extractor.

-   :material-table-large: [Bring your own table](bring_your_own_data.md)

    The scikit-learn estimators and automatic recipe selection.

-   :material-rocket-launch-outline: [Deploy](deployment.md)

    Saved heads, the HTTP service, the tool filter, the confidence gate.

-   :material-database-outline: [Datasets](datasets.md)

    Every public dataset, its pinned source, license and the paper result it backs.

-   :material-flask-outline: [Reproducing the paper](reproducing.md)

    Each paper cell, the command that recomputes it, paper vs released values.

-   :material-code-braces: [Examples](#try-it)

    Five runnable scripts, from the public-benchmark quickstart to cross-agent transfer.

</div>

## How it works

<div class="ta-pipeline" role="img" aria-label="Successful agent trajectories go to TabSchema, LLM-driven feature discovery, which produces a feature card and a compiled extractor; TabSynth adds label-preserving paraphrases; TabHead, CatBoost over the card, returns the decision: a top-k shortlist, a label or a route.">
  <div class="ta-pipeline__step"><strong>successful agent</strong><span>trajectories</span></div>
  <div class="ta-pipeline__arrow" aria-hidden="true">→</div>
  <div class="ta-pipeline__step ta-pipeline__step--key"><strong>TabSchema</strong><span>LLM-driven feature discovery</span></div>
  <div class="ta-pipeline__arrow" aria-hidden="true">→</div>
  <div class="ta-pipeline__step"><strong>feature card</strong><span>+ compiled extractor</span></div>
  <div class="ta-pipeline__arrow" aria-hidden="true">→</div>
  <div class="ta-pipeline__step ta-pipeline__step--key"><strong>TabSynth</strong><span>label-preserving paraphrases</span></div>
  <div class="ta-pipeline__arrow" aria-hidden="true">→</div>
  <div class="ta-pipeline__step ta-pipeline__step--key"><strong>TabHead</strong><span>CatBoost over the card</span></div>
  <div class="ta-pipeline__arrow" aria-hidden="true">→</div>
  <div class="ta-pipeline__step"><strong>decision</strong><span>top-k / label / route</span></div>
</div>

<div class="ta-steps" markdown>

1. **TabSchema** reads your agent's own logs and, with an LLM, proposes behavior-grounded features, writes and
   sandbox-tests the code that extracts them, and keeps only those a judge ensemble accepts. The output is a
   *feature card* plus a reusable, reviewed extractor. It runs once per agent and catalog.
2. **TabSynth** adds deterministic, label-preserving paraphrases of the free-text state where trajectories are
   scarce.
3. **TabHead** scores every candidate of a decision in one pass (CatBoost), with an optional confidence gate that
   defers uncertain decisions back to the original LLM.

</div>

<figure class="ta-figure" markdown="span">
  ![TabSchema: feature analyzer, code extractor and validator, feature aggregator, three judges and a meta-judge produce the feature schema](assets/tabschema.png){ loading=lazy decoding=async }
  <figcaption>TabSchema compiles trajectory-derived schema, state, and dependency signals into tabular features.</figcaption>
</figure>

<figure class="ta-figure" markdown="span">
  ![TabHead: numerical, categorical and textual features plus the tool catalog go into one tabular-textual unit that outputs a ranked shortlist](assets/tabhead.png){ loading=lazy decoding=async }
  <figcaption>TabHead consumes features and candidates to output per-candidate probabilities.</figcaption>
</figure>

## Results at a glance

<div class="ta-results" markdown>

| Decision | Benchmark | Metric | Best baseline | TabAgent | Reproducible here |
|---|---|---|---:|---:|:---:|
| Tool shortlisting | AppWorld, five apps | P@R | 56–92 | **66–96** | traces not released |
| Tool shortlisting | τ²-bench, three domains, 38 tools | P@R | 42 | **78** | ✓ |
| Skill shortlisting | SkillsBench, 195 skills | P@R | 74 | **80** | ✓ |
| Application selection | AppWorld, multi-label | macro&#8209;F1 | 92 | **97** | traces not released |
| Operation selection | Mind2Web, three-way | macro&#8209;F1 | 84 | **92** | ✓ |
| GUI element selection | Mind2Web, three domains | R@3 | 15–23 | **68–74** | ✓ |
| Model routing | LLMRouterBench, 12 models | route@1 | 70 | **71**&nbsp;(on&nbsp;par) | ✓ opt-in download |

</div>

Percentages from the paper's Table 1 and, for AppWorld, its full per-method table. TabAgent is the better of
its real-only and +synth variants. Best baseline is the strongest of BM25, DSR (zero-shot and fine-tuned; on
AppWorld also fine-tuned on the same TabSynth rows, where TabAgent's lead is significant only on Gmail) and the
few-shot LLM in each row; on model routing it is the best single model applied to every task. A fine-tuned 8B small LLM, a reference at about
10²–10⁴× TabAgent's cost, is excluded as in the paper; it scores higher on four of the five AppWorld apps. The
AppWorld rows use the CUGA agent's own trajectories, which are not released. The paper's table also covers the
OAK and BPO enterprise benchmarks, where the small trajectory pools leave TabAgent behind or level with the best
baseline, and sub-agent routing and skill gating, where it is on par with the best fixed default.
[Reproducing the paper](reproducing.md) recomputes every public row.
{ .ta-note }

The reproducible rows run on the feature cards bundled with each public dataset, not on a new TabSchema run, so
reproducing them calls no LLM. A card is *frozen* when its feature list was taken from the paper's experiment
code for that benchmark and fixed, every feature recomputed by the release's code (a few frozen cards are
variants; each card's `provenance` note says how), and *hand-authored* when it was written for the release from
the paper's description of the benchmark's features (`provenance.kind` in each card). The released TabSchema
implementation has not yet been validated against a paper number.
{ .ta-note }

<div class="ta-findings" markdown>

<figure class="ta-figure" markdown="span">
  ![Per-decision latency against shortlisting quality: TabAgent is the fastest and cheapest point, near the top on quality](assets/pareto.png){ loading=lazy decoding=async }
  <figcaption>Per-decision latency (x, log ms) against shortlisting quality P@R (y). TabAgent (star) is the fastest and cheapest point while remaining near the top on quality; filled circles are fine-tuned small LLMs, open circles the same models few-shot. Figure from the paper.</figcaption>
</figure>

<div markdown>

What the paper finds, across three decision families and seven agent benchmarks:

- **Set retrieval.** On AppWorld tool shortlisting, TabAgent recovers essentially all of the tools the GPT-5.4
  shortlister used on its successful runs (Recall@9) at far lower latency and cost.
- **Classification and policy.** It ranks first on the classification decisions and is on par with a strong
  fixed default on the harder routing and gating decisions, where a costly generative router does worse.
- **Representation, not model.** A discriminative head over raw text or embeddings alone collapses; the
  trace-derived features make it accurate.
- **In the loop.** Placed on a live agent's critical path, TabAgent drives AppWorld tasks to completion about
  as often as the GPT-5.4 shortlister it replaces.
- **Few traces.** The head reaches 90% of its full-pool P@R from 118 successful trajectories on AppWorld and
  from 37 to 51 on the four τ²-bench agents.

</div>

</div>

## Install

```bash
pip install tabagent                    # core: tables, TabSynth, TabHead, evaluation
# until the PyPI release: pip install "tabagent @ git+https://github.com/dolev31/TabAgent"
pip install "tabagent[tabschema]"       # + onboarding from agent logs (OpenAI and OpenAI-compatible LLMs)
pip install "tabagent[llm]"             # + Anthropic, Gemini and other LLM providers through LiteLLM
pip install "tabagent[embed]"           # + e5 context-candidate similarity features
pip install "tabagent[serve]"           # + HTTP decision service
```

`pip` resolves the newest dependency releases. For the exact versions pinned in `uv.lock`, clone the
repository and run `uv sync --extra embed`.

## Try it

<!-- named -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolev31/TabAgent/blob/main/notebooks/quickstart.ipynb)
<!-- /named -->

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

<div class="ta-examples" markdown>

| Example | What it shows | Needs |
|---|---|---|
| [01_quickstart_skillsbench.py](https://github.com/dolev31/TabAgent/blob/main/examples/01_quickstart_skillsbench.py) | the paper's components on a public benchmark, cross-validated | CPU; network on first run (3.5 MB of data, 128 MB encoder) |
| [02_bring_your_own_table.py](https://github.com/dolev31/TabAgent/blob/main/examples/02_bring_your_own_table.py) | scikit-learn estimators on your own decision table | CPU, offline |
| [03_onboard_agent_logs.py](https://github.com/dolev31/TabAgent/blob/main/examples/03_onboard_agent_logs.py) | onboarding a decision from your own agent logs | an LLM API key |
| [04_deploy.py](https://github.com/dolev31/TabAgent/blob/main/examples/04_deploy.py) | live decisions in-process, the HTTP service, a tool filter in front of your LLM | a saved head (from 02 or 03) |
| [05_tau2_cross_agent_transfer.py](https://github.com/dolev31/TabAgent/blob/main/examples/05_tau2_cross_agent_transfer.py) | training on one agent's traces and deploying on another's (paper Table 2) | CPU; network on first run (about 600 MB) |

</div>

## Abstract

<div class="ta-abstract" markdown>

Agentic systems chain LLM calls to make closed-set decisions: choosing from a fixed candidate set, as in skill
selection, tool shortlisting, application selection, and routing. These repeated frontier-model calls dominate
per-task latency and cost. We show such decisions need not be made by a language model at all: a compact
discriminative head, trained on the agent's own successful execution traces, can replace the generative one while
leaving the surrounding agent intact. TabAgent has three parts. (i) **TabSchema** extracts schema, state, and
dependency features from trajectories. (ii) **TabSynth** adds schema-aligned synthetic rows where trajectory coverage
is thin. (iii) **TabHead**, a cheap tabular classifier, scores candidates in a single pass. Changing only the label
space, we test TabAgent as a drop-in replacement for the frontier-LLM heads (e.g., GPT-5.4) across three decision
families and seven agent benchmarks. On tool shortlisting it recovers essentially all of the tools the GPT-5.4
shortlister used on its successful runs (Recall@9), at 99.96% lower latency and 99.9996% lower cost per call
(∼10³× and ∼10⁵×). It also ranks first on the classification decisions and is on par with a strong fixed default on
the harder routing and gating decisions at far lower latency and cost, where a costly generative router does worse.
Across these tasks, the gains come from the representation, not the model: a discriminative head over raw text or
embeddings alone collapses, and trace-derived features make it accurate.

</div>

<!-- named -->
<p class="ta-abstract__source">The paper on arXiv: <a href="https://arxiv.org/abs/2602.16429" data-goatcounter-click="paper">abstract page</a> · <a href="https://arxiv.org/pdf/2602.16429" data-goatcounter-click="paper-pdf">PDF</a>. Its first version (v1, February 2026) carries the paper's earlier title, <em>TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers</em>.</p>
<!-- /named -->

## Key terms

<dl class="ta-terms">
<div><dt>Closed-set decision</dt><dd>A decision inside an agent's execution whose answer is selected from a fixed, bounded set of candidates: which skill to invoke, which tool to call next, which application to route to, or which sub-task to gate.</dd></div>
<div><dt>Tool shortlisting</dt><dd>The task of picking a small subset from a catalog of hundreds to thousands of tools, where a tool is any externally callable capability (API endpoint, function, or GUI operation) with a typed schema and observable outputs.</dd></div>
<div><dt>Feature card</dt><dd>The tabular layout TabAgent learns from, with one row per (context, candidate) pair and one column per behavior-grounded feature.</dd></div>
<div><dt>TabSchema</dt><dd>An orchestrated agentic system that proposes features, implements them as executable trace-extraction code, and keeps only those a judge ensemble agrees are decision-relevant.</dd></div>
<div><dt>TabSynth</dt><dd>A schema-aligned augmentation module: a deterministic, rule-based paraphraser that rewrites the free-text state of an existing training row while copying the candidate, the label, and every schema, state, and dependency feature verbatim.</dd></div>
<div><dt>TabHead</dt><dd>The classifier that scores each (context, candidate) pair over the TabSchema feature card in a single pass. It is CatBoost in every main-paper result.</dd></div>
<div><dt>P@R (R-precision)</dt><dd>The share of a task's ground-truth tools that appear in the top R of the ranking, where R is the task's relevant-set size, averaged over tasks. It equals Recall@R and captures completeness.</dd></div>
<div><dt>Recall@k</dt><dd>The share of a task's ground-truth tools recovered in the top k of the ranking, averaged over tasks: recovery at a fixed budget.</dd></div>
</dl>

## Questions

<div class="ta-faq" markdown>

??? question "Can a small classifier replace an LLM for tool selection in an agent?"

    For closed-set decisions an agent makes repeatedly, the paper shows it can. TabAgent trains a CatBoost classifier
    on the agent's own successful traces. On AppWorld tool shortlisting it recovers essentially all of the tools
    GPT-5.4's successful runs used (Recall@9 of 92 to 100 across five apps) at 2.682 ms and $2.0×10⁻⁷ per decision,
    against 7.50 s and $0.052 for the GPT-5.4 shortlister.

??? question "What is TabAgent?"

    TabAgent is a framework that replaces a generative decision head in a running LLM agent, such as a tool
    shortlister or a router, with a compact classifier trained on the agent's own successful execution traces.
    TabSchema turns the traces into a tabular feature card, TabSynth adds paraphrased rows where coverage is thin, and
    TabHead (CatBoost) scores every candidate in one pass. The surrounding agent is left unchanged.

??? question "What exactly does TabAgent learn from?"

    From the agent's own successful execution traces. TabSchema turns them into a tabular feature card: each row is a
    (task context, candidate tool) pair, each column a behavior-grounded feature (for example, the agent's current
    thought, the step index, or whether the candidate has co-occurred with other tools in similar traces), and the
    label is whether the successful trajectory used the candidate. TabAgent learns the decision boundary of the
    generative head from these rows and runs that boundary as a single-pass classifier in place of the LLM.

??? question "Which decisions can TabAgent replace?"

    Closed-set decisions, whose answer comes from a fixed set of candidates. The paper covers three families across
    seven benchmarks: set retrieval (tool and skill shortlisting on AppWorld, τ²-bench, OAK, BPO and SkillsBench),
    classification (application selection on AppWorld, operation and GUI element selection on Mind2Web), and policy
    or reward decisions (sub-agent routing on AppWorld, model routing on LLMRouterBench, and skill gating on
    SkillsBench).

??? question "Does the classifier head matter?"

    Little. With the skill-selection feature card held fixed, heads from classical trees to tabular foundation
    models such as TabPFN and TabSTAR reach useful and comparable P@R on skill selection, and every head collapses
    to P@R 0.10 to 0.21 without the card. The gains come from the trace-derived representation, not the model, which is why one
    lightweight CatBoost head serves every decision family in the paper.

??? question "Does TabSynth always help?"

    Not always. On AppWorld, TabSynth lifts TabAgent P@R by 23% on average, concentrated in domains
    where limited trajectory coverage causes the shortlister to miss required selections (SimpleNote +55%,
    Phone +36%). It is not universally positive: on Spotify it lowers P@R (70 → 61), which the paper attributes to a small
    trajectory pool and strong lexical cues that turn paraphrase diversity into noise rather than coverage. The paper
    therefore applies +synth per card and reports the head that wins.

??? question "How does TabAgent hold up as the number of candidates grows?"

    Its margin widens as the candidate set grows. As the candidate-set size K grows, every LLM adaptation regime
    (generative SFT, cross-encoder, and few-shot) degrades sharply on both skill selection and Mind2Web element
    selection, while the TabAgent classifier follows a shallow power law (R² ≥ 0.97) and stays highest at every
    measured K. TabAgent scores each candidate against fixed trace-derived features, so the per-candidate cost is
    constant and the total cost grows only linearly in the number of candidates, unlike an LLM that must reason over
    the entire option list in-context, whose prompt grows with the catalog and can exceed 100K tokens for the
    closed-set options alone.

??? question "How many trajectories does TabAgent need?"

    On AppWorld, TabHead reaches half of its full-pool P@R from 8 successful trajectories and 90% from 118. On the
    four τ²-bench agents it reaches 90% from 37 to 51. Onboarding therefore costs tens to low hundreds of successful
    traces rather than thousands. With no trajectories for a domain it cannot be applied, since there is nothing to
    mimic.

??? question "What happens when the agent's prompt or base model changes?"

    When the upstream agent's behavior shifts (a new prompt, a new base model) but its trace schema is unchanged,
    only TabHead is refit on the new traces, which takes about 45 seconds on CPU, so drift adaptation does not
    require a re-engineering pass. A change that alters the feature schema itself requires re-running the one-time
    TabSchema and TabSynth onboarding.

??? question "Does replacing the shortlister hurt task success in a live agent?"

    In the paper's live hot-swap on 24 held-out cross-application AppWorld tasks, with the CUGA agent driven by
    GPT-5.4 in both arms, TabAgent reached 27.1% Pass@1 against 29.2% for the GPT-5.4 shortlister. The paired
    difference of -2.1 points has a 95% bootstrap CI of [-18.8, +16.7], which includes zero. The two configurations
    agree on the pass-or-fail verdict for 79% of tasks.

??? question "Does a model trained on one agent's traces work for another agent?"

    Within one architecture, yes. On τ²-bench, four ReAct agents share one scaffold and differ only in the LLM
    backbone. Training on one agent's traces and testing on another's costs about 3.1 points of P@R on average
    (own-agent mean 0.766, cross-agent mean 0.735). The paper states this is transfer within an architecture, not
    evidence that TabAgent is architecture-agnostic.

??? question "When should the LLM still make the decision?"

    When the domain has no trajectory pool, when the agent architecture changes, and when the head is unsure.
    TabHead's confidence flags its own errors at AUROC 0.88, so a confidence gate can defer low-confidence decisions
    to the original LLM. The paper does not claim TabAgent beats the LLM everywhere: the frontier model is retained
    where it still earns its cost.

</div>

<!-- named -->
## Citation

If you use TabAgent, please cite the paper:

<div class="ta-bibtex" data-goatcounter-click="bibtex" data-goatcounter-title="BibTeX" markdown>

```bibtex
@misc{levy2026tabagent,
  title  = {TabAgent: Distilling LLM Agent Components into Cheap Classifiers},
  author = {Levy, Ido and Shlomov, Segev and Yaeli, Avi and Mashkif, Nir and Adi, Asaf and
            Goldstein, Yinon and Shapira, Eilam and Choshen, Leshem},
  year   = {2026},
  eprint = {2602.16429},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2602.16429}
}
```

</div>

<!-- /named -->
