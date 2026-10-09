# On-screen claims and their sources

Every number and claim the explainer shows (video, captions, loop, poster, README thumbnail, OG image, animated
SVG) is listed here with its source. Paper references are to `peo/source/tabagent.tex` of the paper (line numbers of
the version this video was built from, and the LaTeX label of the section). Code claims cite the repository's public
docs. Nothing comes from private data, and the video shows no number that is not below.

## Numbers (all from the paper; AppWorld tool shortlisting, per decision)

| On screen | Exact paper wording | Source |
|---|---|---|
| `~10³× lower latency`, `~10⁵× lower cost` (payoff headline, poster, README thumbnail, OG; caption c14 "about 10³× lower latency and 10⁵× lower cost") | "at 99.96% lower latency and 99.9996% lower cost per call (~10³× and ~10⁵×)" | abstract, line 154; repeated in the conclusion, line 863 |
| `7.50 s` and `$0.052` for the "GPT-5.4 shortlister" | "against 7.50 s and $0.052 for the GPT-5.4 shortlister it replaces" | `subsec:eff_new`, line 654; App. Table `tab:cost-runtime`, line 1949 |
| `2.682 ms` and `$2.0×10⁻⁷` for the "TabAgent head" | "A single TabHead pass runs in 2.682 ms at $2.0×10⁻⁷" | `subsec:eff_new`, line 654; `tab:cost-runtime`, line 1954 |
| "per decision" (eyebrow "AppWorld tool shortlisting · per decision") | the table's caption: "Per-decision cost and runtime for one AppWorld shortlisting decision" | `tab:cost-runtime`, line 1941 |
| "Recovers essentially all of the tools the GPT-5.4 shortlister used on its successful runs (Recall@9)" (payoff card, caption c15) | "On tool shortlisting it recovers essentially all of the tools the GPT-5.4 shortlister used on its successful runs (Recall@9)" | abstract, line 154 |
| "In milliseconds, on a CPU" (c12), "decides in milliseconds", badges "milliseconds" and "on a CPU" | "TabAgent's runtime is on CPU (it needs no GPU)"; 2.682 ms per decision | App. `app:cost_efficiency`, line 1941; line 654 |

The latency and cost bars in the payoff are a race, not a chart: the LLM's bar fills slowly and the head's
"bar" is a dot, because at that scale 2.682 ms of 7.50 s is under one pixel. No scale is claimed.

## Claims in the narration (captions = on-screen text)

| Cue | On screen | Source |
|---|---|---|
| c01 | "Which tool? Which skill? Which model?" | closed-set decisions "as in skill selection, tool shortlisting, application selection, and routing" (abstract, line 152); model routing on LLMRouterBench (Table `tab:master_new`) |
| c02 | "Agents ask a frontier LLM, again and again. Slow and costly." | "Agentic systems repeatedly call an LLM ... to make closed-set decisions, which are both expensive and slow" (Introduction, line 256); "These repeated frontier-model calls dominate per-task latency and cost" (abstract, line 152) |
| c03 | "But the agent's successful runs already hold the answers." | "trained on the agent's own successful execution traces" (abstract, line 153); labels "derive[d] ... from successful trajectories" (`sec:setup`, line 373) |
| c04 | "So turn each decision into a table, and train a classifier." | "we cast each decision's snapshot into a textual-tabular representation, letting a cheap, non-generative discriminator read the live state" (Introduction, line 259) |
| c05 | "LLM agents read the successful runs and propose features." | TabSchema is "an orchestrated agentic system that proposes features" from execution traces (`sec:tabschema`, line 308); README "How it works" (line 130) |
| c06 | "They write extraction code, run it in a sandbox, and validate it." | "an executor (writes Python extraction code in a sandbox), a validator (an LLM-as-a-judge pass that checks the code ...)" (`sec:tabschema`, line 308); docs/components.md lines 11-14 |
| c07 | "Three judges score each feature. A MetaJudge decides." | "Three judges, not one or five"; "The MetaJudge receives the three judge scores ... issues a single explained decision (accept, conditionally accept, or reject)" (`app:tabschema:engineering`, lines 1227 and 1231); docs/components.md lines 16-17 |
| c08 | "Out comes a feature card: one row per candidate." | "a feature card, with one row per (context, candidate) pair and one column per behavior-grounded feature" (Sec. 3, line 302) |
| c09 | "TabSynth paraphrases the free text to grow the training set." | "it rewrites the free-text state of an existing (context, candidate) training row" and "expands supervision" (`sec:tabsynth`, lines 316-318) |
| c10 | "Candidate and label are copied verbatim, so it never invents a positive." | "while copying the candidate, the label, and every schema, state, and dependency feature verbatim"; "cannot fabricate a positive" (`sec:tabsynth`, lines 318-319) |
| c11 | "TabHead, a CatBoost classifier, ranks every candidate in one pass." | "TabHead, a cheap tabular classifier, scores candidates in a single pass" (abstract, line 153); "TabHead is CatBoost in every main-paper result" (TabHead section, line 323); "at inference we sort all candidates by p(y given x) and return the top-K" (line 325) |
| c12 | "In milliseconds, on a CPU." | see Numbers above |
| c13 | "Unsure? A confidence gate hands the decision back to the LLM." | "low-confidence decisions route to the original LLM shortlister" (TabHead section, line 325); README line 135: "an optional confidence gate that defers uncertain decisions back to the original LLM" |
| c14, c15 | the payoff | see Numbers above |
| c16 | "TabAgent: distilling LLM agent components into cheap classifiers." | the paper's title (line 136) |
| c17 | "Onboard your own agent from its logs." + the repository URL | README "Three ways to use it", lines 190-198 (`tabagent onboard ./agent_logs --decision ToolShortlist`) |

## Labels and diagrams

| On screen | Source |
|---|---|
| Phase subtitles "discovers features", "augments the rows", "decides in milliseconds" | the brand spec's three phases (TabSchema discovers features; TabSynth label-preserving paraphrase augmentation; TabHead decides in milliseconds); grounded in the rows above |
| TabSchema stages: "Analyze / propose features", "Write code / extraction functions", "Validate / sandbox and checks", "Judge / 3 judges + MetaJudge" | `sec:tabschema` line 308; docs/components.md lines 9-17 |
| Validation checks "runs in a sandbox", "reads only real fields", "no outcome fields" | docs/components.md line 14: the CodeValidator's "deterministic checks for hallucinated fields, outcome fields, forbidden calls and imports"; paper line 1229 (fields "that do not exist in the agent's logging schema") |
| Judge names "Relevance", "Generality", "Impact & Uniqueness", "MetaJudge" | docs/components.md line 16 |
| Function names `extract_features_for_classification` and `extract_candidate_features` | docs/components.md lines 55 and 132 |
| Feature-card groups "per-decision features", "per-candidate features", "label" | per-candidate features: docs/components.md, "Per-candidate features" (line 132 and the examples on lines 151-153); the label is "whether the successful trajectory used the candidate" (Introduction, line 261) |
| Loop and poster line "Replace an LLM agent's closed-set decisions with a cheap classifier." and "Trained on the agent's own successful runs, with the surrounding agent left unchanged." | README lines 12 and 48 |
| Poster/OG: "per decision than the GPT-5.4 shortlister it replaces (AppWorld tool shortlisting)" | line 654 |
| "Ranked shortlist / in one classifier pass" (loop, SVG) | line 654 ("one classifier pass"), line 325 (top-K) |
| "TabAgent", the mark and the wordmark | brand spec |
| Paper title "Distilling LLM Agent Components into Cheap Classifiers" | line 136 |
| `pip install "tabagent[tabschema]"` | README line 152 ("+ onboarding from agent logs"). The plain `pip install tabagent` installs the core only; `tabagent onboard` needs the `tabschema` extra (it imports `openai`). The command is the `install_cmd` constant in `config.json`. |
| `tabagent onboard ./logs --decision ToolShortlist` | README line 197; docs/onboarding.md lines 9-10. The `onboard_cmd` constant in `config.json`. |
| `github.com/dolev31/TabAgent` | the `repo_url` constant in `config.json` (the public name is not final) |
| `arXiv 2602.16429` | README badge and citation (the `arxiv_id` constant in `config.json`) |

## Illustrative content (not claims)

These are example data drawn to show the mechanism; they carry no measured value.

- The running example, a user asking to return a roommate's order on Amazon, and its fields (`intent` text
  "My roommate asked me to return something I had ordered for them and venmo them the money for it.",
  `step_id` 24, `app_name` amazon, the thought "Extraction is complete ...", the candidates `show_orders`,
  `initiate_return`, `show_returns`) come from the paper's example feature card (App.
  `app:features_card_example`, lines 2125-2161). The other tool names (`search_products`, `add_to_cart`,
  `cancel_order`, `write_review`) are made up for the illustration.
- Feature names: `intent`, `step_id`, `app_name` are fields of the paper's example card; `mentioned_by_user` and
  `args_in_context` name two of the docs' examples of per-candidate features (docs/components.md lines 151-153);
  `raw_log_length` is a made-up feature for the judges to reject. Their 0/1 values are made up.
- The paraphrase "Please handle the following." is the paper's own TabSynth example framing (App.
  `app:TabSynth`, line 2119); "Task:" is a made-up second framing.
- The extraction code in the editor is illustrative, not code TabSchema produced.
- Judge score dots, the confidence meter, the ranked bars and the training-set bar carry no numbers and stand for
  no measured values.
