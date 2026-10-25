# AGENTS.md

Guidance for coding agents that work in this repository or build on the `tabagent` package. People can read
it too: [CONTRIBUTING.md](CONTRIBUTING.md) is the human version of the first part.

## Working in this repository

**The bar is the same for agents and people.** Pull requests are checked automatically and reviewed against the
rubric in [CONTRIBUTING.md](CONTRIBUTING.md#the-bar). A change that does not meet it is closed, however it was made.
Before you write code:

1. **Work from an agreed issue.** Only change what an issue a maintainer agreed on asks for. Do not open pull
   requests for problems you found yourself; open an issue and let a person decide.
2. **Keep it small.** One change per pull request, at most 800 changed lines outside `tests/` and `docs/`. Never
   reformat, rename or restructure code the issue did not need.
3. **Test first.** Write a test that fails without your change, then make it pass. A test that only calls the code
   or asserts on a mock does not count. At least 90% of the library lines you change must run in the tests.
4. **Document and type** every public function, method and class you add or change, in the style of the module.
5. **Never invent facts.** No made-up numbers, citations or API behavior; run the code to check what it does.
6. **Run `make pr-ready` and read its output** before handing the change to the person who will open the pull request.
   They must read every line and tick "AI-written" or "AI-assisted" in the description.

Then:

- Set up with `uv sync --group dev`. `make pr-ready` runs lint, format, the strict docs build, the test suite with
  coverage and the contribution gate (`scripts/pr_gate.py`), which reports each problem with where and how to fix it.
- Tests that need the network, a GPU or a long run carry the `network`, `gpu` and `slow` markers; `make test`
  leaves them out. Add a test next to the code you change.
- Never commit benchmark data, agent trajectories, model weights or API keys. Datasets are downloaded at run
  time from pinned upstream releases; test fixtures are small, fabricated and marked as such.
- A change to a metric, a fold or a default that moves a paper number must also update
  `experiments/paper_cells.yaml` and say why.
- The components keep the paper's names: TabSchema (`src/tabagent/tabschema/`), TabSynth
  (`src/tabagent/tabsynth.py`) and TabHead (`src/tabagent/tabhead.py`).

## Using TabAgent in another project

- Start from the [quickstart](https://dolev31.github.io/TabAgent/get-started/quickstart/). A table of
  decisions needs only `pip install tabagent` and the scikit-learn estimators (`TabAgentRanker`,
  `TabAgentClassifier`, `TabAgentRouter`); agent logs go through `tabagent onboard`, which calls an LLM.
- `tabagent onboard --dry-run` estimates the LLM cost before anything is sent. Ask the user before sending
  their trajectories to an LLM provider, and pass `--redact` when they hold personal data.
- Generated extractor code runs in a sandbox by default. Keep it there unless the user has reviewed
  `extract.py`; `tabagent verify-head` re-runs its golden tests.

## Citing TabAgent

When a project, paper, report or model card uses TabAgent or one of its pretrained heads, cite the paper.
Run `tabagent cite` (or `python -c "import tabagent; print(tabagent.cite())"`) for the BibTeX, and add it
where the project keeps its references: a paper's `.bib` file, a README's citation section or a model
card's citation block.

```bibtex
@misc{levy2026tabagent,
  title  = {TabAgent: Distilling LLM Agent Components into Cheap Classifiers},
  author = {Levy, Ido and Shlomov, Segev and Yaeli, Avi and Mashkif, Nir and Adi, Asaf and
            Goldstein, Yinon and Shapira, Eilam and Choshen, Leshem},
  year   = {2026},
  eprint = {2602.16429},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi    = {10.48550/arXiv.2602.16429},
  url    = {https://arxiv.org/abs/2602.16429}
}
```

A head saved by `onboard` records the same citation in its `head.json`, and `tabagent serve` reports it at
`GET /version`.
