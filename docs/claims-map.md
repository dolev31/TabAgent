---
search:
  exclude: true
---

# Claims map

Every number on the [README](https://github.com/dolev31/TabAgent#readme), the [landing page](index.md) and the social
images (`docs/assets/social-card.png`, `docs/assets/github-social-preview.png`, `docs/assets/pipeline.svg`), with its
source. Paper numbers come from the paper's text (section, table or figure; "line" is the line of the paper's LaTeX
source where the sentence sits). Code claims come from the repository's own documentation or from a run of the
released code. Nothing here comes from private data. Keep this page in step when a number changes on either surface.
The explainer video, its poster and the README's video thumbnail (`docs/assets/explainer-poster.webp`,
`docs/assets/explainer-thumb.webp`) list their claims in
[media/explainer/claims.md](https://github.com/dolev31/TabAgent/blob/main/media/explainer/claims.md).

## Paper numbers

| Number on the page | Where it appears | Source in the paper |
|---|---|---|
| 2.682 ms per decision (TabHead) | README Why, pipeline image, landing stat card, social images, FAQ | §5 efficiency paragraph (line 654); App. cost table and "How the TabHead per-decision cost is computed" (lines 1941–1960) |
| $2.0×10⁻⁷ per decision (TabHead) | same | same |
| 7.50 s, $0.052 per decision (GPT-5.4 shortlister) | same | same; App. cost table row "CUGA's GPT-5.4 (API) shortlister" (line 1949) |
| more than 10³× lower latency, 10⁵× lower cost | README Why, landing stat cards | §5 efficiency paragraph (line 654); §3.3 TabHead (line 325) |
| 99.96% lower latency, 99.9996% lower cost (~10³×, ~10⁵×) | landing Abstract; ~10³×, ~10⁵× on the explainer poster and thumbnail | Abstract (line 154); Conclusion (line 863) |
| Recall@9 of 92–100 across five AppWorld apps | README Why, landing stat card, FAQ | §5 set retrieval (line 598); Table 1 R@k column (97, 95, 100, 100, 92) and App. Table `perf_new` |
| three decision families, seven agent benchmarks | landing findings, Abstract, FAQ | Abstract (line 154); §1 (line 264) |
| Tool shortlisting, AppWorld, five apps: best baseline 56–92, TabAgent 66–96 (P@R) | README and landing results table | App. Table `perf_new`: per app the best of BM25, DSR zero-shot, DSR-FT real-only and real+synth, few-shot Llama (69, 56, 85, 92, 67); TabAgent the better of real-only and +synth (71, 66, 90, 96, 70) |
| Tool shortlisting, τ²-bench, three domains, 38 tools: 42 vs 78 | results tables | Table 1 row "combined" (DSR-FT 42; TabAgent 77, +synth 78); merged 38-tool catalog (line 601) |
| Skill shortlisting, SkillsBench, 195 skills: 74 vs 80 | results tables | Table 1 "Skill selection" (DSR-FT 74, TabAgent 80); 195 candidate skills (line 629) |
| Application selection, AppWorld: macro-F1 92 vs 97 | results tables | Table 1 (DSR-FT 92, TabAgent 97) |
| Operation selection, Mind2Web, three-way: macro-F1 84 vs 92 | results tables | Table 1 (DSR-FT 84, TabAgent 92) |
| GUI element selection, Mind2Web, three domains: R@3 15–23 vs 68–74 | results tables | Table 1 (best baselines 15, 18, 23; TabAgent +synth 68, 74, 72) |
| Model routing, LLMRouterBench, 12 models: route@1 70 vs 71 (on par) | results tables | Table 1 (best default 70, TabAgent 71, "=" within the paired 95% CI); 12 models (line 642) |
| TabAgent's lead over DSR fine-tuned on TabSynth rows is significant only on Gmail | results notes | §5 (line 598) |
| fine-tuned 8B small LLM at about 10²–10⁴× TabAgent's cost; higher on four of five AppWorld apps | results notes | Table 1 caption; Table 1 FT-SLM column (86, 81, 91, 100, 67 against 71, 66, 90, 96, 70) |
| 118 successful trajectories reach 90% of full-pool P@R on AppWorld; 37 to 51 on the four τ²-bench agents; half from 8 | landing stat card and findings, README More findings, FAQ | §5 (line 656) |
| live hot-swap on 24 held-out AppWorld tasks: 27.1% vs 29.2%, difference −2.1, 95% CI [−18.8, +16.7] | README More findings, FAQ | §5 in-the-loop (line 666), Table `inloop_main` (lines 679–680) |
| the two configurations agree on 79% of tasks | FAQ | §5 (line 668) |
| about 3.1 points of P@R across agents (own-agent mean 0.766, cross-agent mean 0.735) | README More findings, FAQ | §1 (line 266); §5 (line 601), Table 2 |
| heads collapse to P@R 0.10–0.21 without the card | README More findings, FAQ | §5 (line 645), Fig. 3 caption |
| power law with R² ≥ 0.97 as the candidate set grows | FAQ | Fig. 4 caption (line 634) |
| prompts that can exceed 100K tokens for the options alone | FAQ | §5 (line 629) |
| TabSynth lifts P@R by 23% on average (SimpleNote +55%, Phone +36%); Spotify 70 → 61 | FAQ | §5 (line 598); App. A, "Where synthesis backfires" (line 890) |
| refitting TabHead takes about 45 seconds on CPU | FAQ | §3.3 (line 325) |
| confidence flags its own errors at AUROC 0.88 | FAQ | §3.3 (line 325); §5 (line 688); App. A (line 888); App. error analysis (line 2370) |

## Code claims (the repository's documentation, or a run of the released code)

| Claim | Where it appears | Source |
|---|---|---|
| `examples/01_quickstart_skillsbench.py` prints P@R ≈ 0.80 and Recall@9 ≈ 0.90 | README Quick start, landing Reproduce tab | run on 2026-10-08 from a fresh install of this branch: P@R 0.796, Recall@9 0.905 (87 tasks); [Reproducing the paper](reproducing.md) |
| it downloads 3.5 MB of data and a 128 MB encoder, in a few minutes on a laptop CPU | README Quick start, landing Reproduce tab | [Datasets](datasets.md) (SkillsBench 3.5 MB); the example's own docstring; a fresh install and run took 5 min 47 s, downloads included |
| Python 3.10 to 3.13 | README badge, landing pip tab | `pyproject.toml` classifiers and the CI matrix |
| `pip install tabagent` and the extras `tabschema`, `llm`, `embed`, `serve` | README, landing | `pyproject.toml` optional dependencies; the package is published to PyPI at launch |
| `tabagent scan`, `onboard --dry-run`, `onboard --out`, `serve`, `verify-head`, `check-logs`, `components` | README, landing CLI tab | `tabagent --help`, each subcommand run with `--help` on 2026-10-08 |
| log formats: OpenAI, Anthropic, LangChain, τ²-bench, CUGA, generic JSON | README, landing | [Onboard an agent](guides/check-and-debug-logs.md#log-formats) |
| `tabagent serve`: `POST /decide`, warm sandboxed workers, deferral on late extraction, readiness, Prometheus metrics, drift monitoring | README, landing | [Deploy](guides/production-runbook.md) |
| `ToolFilter` sends only the shortlisted tools, and the full list when the head defers | README, landing | [Deploy](guides/deploy-and-serve.md#live-prefixes) |
| `TabAgentRanker` infers the card and picks TabSynth and the prior by nested validation | README, landing | [Bring your own table](guides/bring-your-own-table.md); `examples/02_bring_your_own_table.py`, run on 2026-10-08 |
