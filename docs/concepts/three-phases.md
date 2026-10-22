---
title: "The three phases"
description: "TabSchema discovers features from an agent's trajectories, TabSynth adds label-preserving paraphrases, and TabHead scores every candidate in one pass: how each phase works."
lead: "TabSchema discovers features, TabSynth grows the training rows where trajectories are scarce, TabHead decides."
kind: Concept
---

# The three phases

Each phase runs once per agent and catalog. At inference the head replaces the LLM call; the rest of the agent is unchanged. The release's additions to TabSchema around per-candidate features are on their own page: [Per-candidate features](per-candidate-features.md).

## TabSchema: feature discovery from traces

`TabSchema(llm).discover(trajectories, target)` runs the paper's FeatureOrchestrator (Sec. 3.1, App. D):

1. **FeatureAnalyzer** proposes features from sampled decision points (pre-decision prefix only). The sample
   (`n_analyze`, 20 by default) takes each task once before repeating one, and covers the onboarding data's
   sources, positions in the run, prefix lengths and decision labels (`sample_strategy="coverage"`, the
   release's choice; the paper does not say how its decision points were sampled).
2. **CodeExtractor → CodeExecutor** write extraction code and run it in a sandbox, retrying up to three times
   with the error and the failed code in the prompt.
3. **CodeValidator** checks that the executed code reads real fields and implements the plan (an LLM judge plus
   deterministic checks for hallucinated fields, outcome fields, forbidden calls and imports).
4. **FeatureAggregator** merges the proposals of all decision points.
5. **Relevance, Generality, Impact & Uniqueness judges** score each feature (1–5) concurrently; the
   **MetaJudge** accepts, conditionally accepts or rejects it. Accepted and conditionally accepted features go
   on (the paper does not say which outcomes enter the card; the release keeps both because the conditional
   ones clearly helped the head in its experiments), and `TabSchemaConfig(accept=("accept",))` keeps only
   accepted ones.
6. **Compile**: one reusable extractor for the accepted features, run and quality-checked on every onboarding
   decision point; the result is a feature card plus `extract.py`. The library appends two fixed, non-LLM
   functions: the decision's context as text (`context_text`), so every card can match the context against the
   candidates' text, and the text of the prefix's last step (`last_step_text`, below).

The release adds four checks of the values the compiled extractor computes. **They are not in the paper**, and
each can be turned off in `TabSchemaConfig` ([details](../guides/verify-generated-code.md)):

* **Verification** (`tabagent.tabschema.verify`, in the compile step): determinism on every decision point
  (fresh interpreters, shuffled order) and invariance to the run's outcome fields; columns that fail are
  dropped. Invariance of the adapter's prefixes to the run's future (rebuilt from changed runs): a failure stops
  onboarding. Robustness on short or incomplete prefixes is only reported.
* **OutputAuditor** (`tabagent.tabschema.audit`, stage `auditor`): an LLM judges each feature's value on 12
  coverage-sampled decision points as correct, incorrect or cannot tell; a feature is dropped only when its
  incorrect rate is confidently above the threshold (a Wilson lower bound). Its prompt, `output_auditor.txt`,
  is the release's.
* **Usefulness** (`tabagent.tabschema.usefulness`): drop-feature importance (a feature's columns dropped
  together, plus every LLM-written feature at once) by task-grouped cross-validation on the onboarding rows,
  CAAFE-style (Hollmann et al., 2023), with Holm-adjusted paired t-test p-values. It is a report, or prunes
  features harmful under two seeds with `prune_harmful=True`.
* **Golden tests** (`extractor_tests.json`, saved with the head): `tabagent verify-head` re-runs them before
  deployment.

The seven prompts the paper prints (FeatureAnalyzer, CodeExtractor, FeatureAggregator, the three judges and the
MetaJudge, App. D.2 and D.3) are reproduced verbatim and hash-pinned in the tests. The paper prints no
CodeValidator prompt: the release wrote one from the validator's description (App. D.1). The OutputAuditor's
prompt is the release's too, for a stage the paper does not have; both are hash-pinned as well. The paper describes the
CodeExecutor's retries but not the text added to the CodeExtractor prompt after a failed attempt; the wording of
that retry block is the release's too. The release's other additions are sections appended after a rendered
prompt, never inside it (`prompts.ADDITIONS`, hash-pinned): the CodeExtractor is asked to emit each text feature
once, as plain text, and, for a decision with candidates, told about the optional per-candidate function
described next, which the CodeValidator and the OutputAuditor are told about too; the FeatureAnalyzer is told
that it may propose features of a candidate, and the FeatureAggregator, judges and MetaJudge that such
features are obtainable. That function keeps the
paper's entry point, `extract_features_for_classification(trajectory_data)`, and writes the paper's examples of
features that relate the context to a candidate as stateless code over the prefix and one candidate.
Every LLM call is cached in a run directory (a resumed run costs
nothing), priced by a cost meter with a budget stop, and the extractor runs in a subprocess sandbox with an empty
environment (`DockerSandbox` adds no network and a read-only filesystem; see `SECURITY.md`).

The paper ran TabSchema with GPT-5.4 and found open models unreliable for this loop. The release calls OpenAI,
Azure OpenAI, and any OpenAI-compatible server through `OPENAI_BASE_URL`, and Anthropic, Gemini and others
through LiteLLM (`pip install "tabagent[llm]"`). The reproduced paper numbers do not run TabSchema: they use the
frozen or hand-authored cards bundled with the public datasets ([Reproducing the paper](../reproducing.md)), and
the released TabSchema has not yet been validated against a paper number. One public onboarding with an LLM
compares versions of the release's own pipeline (Evidence, under per-candidate features below).

## TabSynth: label-preserving augmentation

`TabSynth(k)` implements Algorithm 1 (App. H): for each real training row, `k` deterministic rule-based
paraphrases of the free-text state columns, with the candidate, the label and every other feature copied
verbatim, and duplicates of real rows dropped. It never creates a new (context, candidate) relationship.
Use it where it helps: the automatic recipe selection adopts it only when nested validation says so.

The extended operators the paper adds on τ²-bench (App. H) are options of the same class:
`recompute=True` builds each variant from the raw row with the paraphrased text and applies the fitted card,
so text-derived features (length, lexical and embedding similarity) follow the paraphrase; `entity_swap=True`
replaces names, identifiers, codes, contact details and amounts with other well-formed values before
paraphrasing; `allocation="inverse_sqrt_frequency"` gives each decision variants in proportion to `1/sqrt(f)`
of its rarest relevant candidate (mean `k`, between 1 and 12). The τ² cells choose among no synthesis and four
of these configurations by a task-grouped 4-fold inner cross-validation on each training fold.

## TabHead: the classifier

`TabHead(kind, preset=...)` is CatBoost over a fold-fitted featurizer (TF-IDF for text, codes for categoricals,
medians for numbers). A card with more than six text columns (an LLM-written extractor can return one text in
several forms) gets a dense text representation instead (`text_repr="auto"`, the release's addition): TF-IDF
over each column's distinct texts reduced by truncated SVD to 64 dimensions. In one internal API-shortlisting
run with many text columns (no interval) it improved ranking over per-column TF-IDF, in less than half the
time; the automatic recipe search also tries the other representation and keeps it when it scores strictly
better.
Cards with fewer text columns, the paper's among them, keep TF-IDF. The default preset, `paper`, is the one configuration the paper uses for every decision
(Sec. 4.2: 300 iterations, learning rate 0.08, depth 6, CatBoost's defaults otherwise); `catboost-default` is
CatBoost's own schedule (1000 iterations, automatic learning rate). Any sklearn-style classifier plugs in with
`estimator=` (the head ablation of Fig. 3 and App. K.3).

`confidence(...)` gives the paper's gate signal (the margin at R, or the cutoff-free largest score gap; App.
N.3); `TabAgent.decide(..., defer_below=τ, fallback=llm)` defers uncertain decisions to the original LLM.
