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
   **MetaJudge** accepts, conditionally accepts or rejects it. Only accepted features go on: conditionally
   accepted ones are excluded by default, and `TabSchemaConfig(accept=("accept", "conditional"))` includes them.
6. **Compile**: one reusable extractor for the accepted features, run and quality-checked on every onboarding
   decision point; the result is a feature card plus `extract.py`. The library appends a fixed, non-LLM
   function that adds the decision's context as text (`context_text`), so every card can match the context
   against the candidates' text.

The seven prompts the paper prints (FeatureAnalyzer, CodeExtractor, FeatureAggregator, the three judges and the
MetaJudge, App. D.2 and D.3) are reproduced verbatim and hash-pinned in the tests. The paper prints no
CodeValidator prompt: the release wrote one from the validator's description (App. D.1). The paper describes the
CodeExecutor's retries but not the text added to the CodeExtractor prompt after a failed attempt; the wording of
that retry block is the release's too. Every LLM call is cached in a run directory (a resumed run costs
nothing), priced by a cost meter with a budget stop, and the extractor runs in a subprocess sandbox with an empty
environment (`DockerSandbox` adds no network and a read-only filesystem; see `SECURITY.md`).

The paper ran TabSchema with GPT-5.4 and found open models unreliable for this loop. The release calls OpenAI,
Azure OpenAI, and any OpenAI-compatible server through `OPENAI_BASE_URL`, and Anthropic, Gemini and others
through LiteLLM (`pip install "tabagent[llm]"`). The reproduced paper numbers do not run TabSchema: they use the
frozen or hand-authored cards bundled with the public datasets ([Reproducing the paper](reproducing.md)), and
the released TabSchema has not yet been validated against a paper number.

**Dependency features** are not code the LLM writes: a card references the library's fitted statistics
(`fitted:<name>`), which are fitted on the training rows of each fold. They are the candidate prior
(`candidate_prior`), the context-candidate embedding similarity (`embedding_cosine`), the outcome of each
candidate on the most similar training tasks (`knn_outcome`) and co-usage (`co_usage`): for a column listing the
candidates used before the decision (an extractor feature with the tools the agent called earlier in the run, for
example), how often the training runs chose the candidate after them, smoothed toward its prior.

**Discovery inside each fold.** `onboard` discovers the card once, on all onboarding trajectories, so the
cross-validation in its report estimates the head on a fixed card. `tabagent.cross_validate_discovery(logs,
target, ...)` runs `discover` on each training fold's trajectories only, applies that fold's extractor to its
held-out decisions and reports the usual cross-validated metrics, at the LLM cost of one discovery per fold (one
budget covers all folds; each fold's run directory is cached, so a rerun is free).

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
