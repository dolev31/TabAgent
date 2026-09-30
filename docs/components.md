# TabSchema, TabSynth, TabHead

## TabSchema: feature discovery from traces

`TabSchema(llm).discover(trajectories, target)` runs the paper's FeatureOrchestrator (Sec. 3.1, App. D):

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

The seven prompts reproduce the paper's App. D.2 and D.3 verbatim and are hash-pinned in the tests. Every LLM call is
cached in a run directory (a resumed run costs nothing), priced by a cost meter with a budget stop, and the
extractor runs in a subprocess sandbox with an empty environment. The paper ran TabSchema with GPT-5.4; any
OpenAI-compatible endpoint or LiteLLM provider works (the paper found open models unreliable for this loop).

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
medians for numbers). The default preset, `paper`, is the one configuration the paper uses for every decision
(Sec. 4.2: 300 iterations, learning rate 0.08, depth 6, CatBoost's defaults otherwise); `catboost-default` is
CatBoost's own schedule (1000 iterations, automatic learning rate). Any sklearn-style classifier plugs in with
`estimator=` (the head ablation of Fig. 3 and App. K.3).

`confidence(...)` gives the paper's gate signal (the margin at R, or the cutoff-free largest score gap; App.
N.3); `TabAgent.decide(..., defer_below=τ, fallback=llm)` defers uncertain decisions to the original LLM.
