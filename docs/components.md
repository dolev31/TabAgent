# TabSchema, TabSynth, TabHead

## TabSchema: feature discovery from traces

`TabSchema(llm).discover(trajectories, target)` runs the paper's FeatureOrchestrator (Sec. 3.1, App. C):

1. **FeatureAnalyzer** proposes features from sampled decision points (pre-decision prefix only).
2. **CodeExtractor → CodeExecutor** write extraction code and run it in a sandbox, retrying up to three times
   with the error and the failed code in the prompt.
3. **CodeValidator** checks that the executed code reads real fields and implements the plan (an LLM judge plus
   deterministic checks for hallucinated fields, outcome fields, forbidden calls and imports).
4. **FeatureAggregator** merges the proposals of all decision points.
5. **Relevance, Generality, Impact & Uniqueness judges** score each feature (1–5) concurrently; the
   **MetaJudge** accepts, conditionally accepts or rejects it.
6. **Compile**: one reusable extractor for the accepted features, run and quality-checked on every onboarding
   decision point; the result is a feature card plus `extract.py`.

The seven prompts reproduce the paper's App. C verbatim and are hash-pinned in the tests. Every LLM call is
cached in a run directory (a resumed run costs nothing), priced by a cost meter with a budget stop, and the
extractor runs in a subprocess sandbox with an empty environment. The paper ran TabSchema with GPT-5.4; any
OpenAI-compatible endpoint or LiteLLM provider works (the paper found open models unreliable for this loop).

## TabSynth: label-preserving augmentation

`TabSynth(k)` implements Algorithm 1 (App. G): for each real training row, `k` deterministic rule-based
paraphrases of the free-text state columns, with the candidate, the label and every other feature copied
verbatim, and duplicates of real rows dropped. It never creates a new (context, candidate) relationship.
Use it where it helps: the automatic recipe selection adopts it only when nested validation says so.

## TabHead: the classifier

`TabHead(kind, preset=...)` is CatBoost over a fold-fitted featurizer (TF-IDF for text, codes for categoricals,
medians for numbers). Two presets exist: `paper` (the configuration stated in the paper: 1000 iterations,
depth 6, automatic learning rate) and `tmlr-runs` (300 iterations, learning rate 0.08: the configuration that
produced the reported numbers). Any sklearn-style classifier plugs in with `estimator=`.

`confidence(...)` gives the paper's gate signal (the margin at R, or the cutoff-free largest score gap);
`TabAgent.decide(..., defer_below=τ, fallback=llm)` defers uncertain decisions to the original LLM.
