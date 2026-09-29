# TabAgent

**Distill an LLM agent's closed-set decisions into cheap tabular classifiers.**

> Work in progress. This repository accompanies the paper *TabAgent: Distilling LLM Agent Components into
> Cheap Classifiers*. The full README, quickstart and reproduction guide land with the first release.

TabAgent replaces repeated frontier-LLM calls for closed-set decisions (tool shortlisting, skill selection,
application selection, routing, gating) with a compact classifier trained on the agent's own successful
trajectories:

1. **TabSchema** extracts schema, state and dependency features from trajectories.
2. **TabSynth** adds schema-aligned, label-preserving synthetic rows where coverage is thin.
3. **TabHead** scores every candidate in a single pass.

## Development

```bash
make setup    # uv sync --group dev + pre-commit hooks
make check    # lint + tests + privacy guard
```

## License

Apache-2.0. Benchmark data is never redistributed; see [NOTICE](NOTICE).
