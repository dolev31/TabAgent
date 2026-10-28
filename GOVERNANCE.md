# Governance

Short, because the project is small and pretending otherwise would be worse than saying so.

## Who decides

TabAgent is maintained by Ido Levy ([@dolev31](https://github.com/dolev31)), who has final say on what merges. That
describes where the project is, not a permanent structure: when a second person reviews regularly and their judgment
is good, they get commit rights, and this file changes.

Deciding what merges is not deciding what is true. Anything that touches a number is settled by evidence.

## Decisions about correctness

**A number is not settled by seniority.** If you can show that a metric, a fold, a leakage guard or a reproduced
result is wrong, that ends the argument, whoever wrote the code. The bar is a failing test or a command anyone can
run, not an opinion. The [CHANGELOG](CHANGELOG.md) records every result that changed because of one.

## Result disputes

When a number in the paper or the docs does not reproduce, open a
[result dispute](https://github.com/dolev31/TabAgent/issues/new?template=result_dispute.yml) with the command, its
output and `uv pip freeze`. Most resolve without a ruling: [docs/reproducing.md](docs/reproducing.md) records each
cell's command, its pinned data and the value it gives, and the two runs usually differ in one of them (CatBoost
fits on the Mind2Web element row, for one, move by up to about 0.8 points between x86-64 and arm64).

When a genuine defect is found, the release notes and `experiments/paper_cells.yaml` say which value changed and why.
**Nothing is quietly rewritten**: the paper's value stays beside the corrected one.

## Breaking changes

Before 1.0, breaking changes ship when they make the library more correct, with the reason in the
[CHANGELOG](CHANGELOG.md). Four things get a deprecation period of at least one minor release instead of a straight
break, because they invalidate other people's work:

- **The saved-head format** (`head.json`, `manifest.json`, the card): heads people trained and shared must keep
  loading. A new format version reads the old one.
- **Metric definitions** (`tabagent.evaluation`): a change alters what every reported number means.
- **The serving API** of `tabagent serve` (`/decide`, `/decide_batch`, `/version`, ...), which agents call.
- **The scikit-learn estimators' parameters**, which pipelines and grid searches are written against.

## Scope

TabAgent replaces an LLM agent's repeated closed-set decisions with a classifier trained on the agent's own successful
runs. A proposal is weighed by whether it makes that more accurate, safer, cheaper or easier to adopt. Free-text
generation, planning, and hosted onboarding are out of scope.

## Conduct

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) applies everywhere, including code review. Enforcement is the maintainer's
responsibility.
