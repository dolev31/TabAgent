# Bring your own decision table

If your decisions are already a table, use the scikit-learn estimators.

```python
import tabagent
from tabagent import TabAgentRanker

rows = tabagent.pointwise(decisions, chosen="tools_used", catalog=tools_df, candidate_col="tool")
X, y = rows.drop(columns=["chosen"]), rows["chosen"]
ranker = TabAgentRanker(group_col="task_id", candidate_col="tool").fit(X, y)
ranker.rank(X_new, k=5)
print(ranker.report(X, y))
```

* `TabAgentRanker`: ranking; `predict` does multi-label decoding (threshold + backoff).
* `TabAgentClassifier`: single-label decisions (a standard sklearn classifier).
* `TabAgentRouter`: routing; `route(X)` returns the chosen arm per decision; `score` is route@1.

What `fit` does: infers a feature card from your columns (types; context vs candidate-side text; a
context–candidate lexical overlap and, with `tabagent[embed]`, an e5 similarity; an in-fold candidate prior),
then selects TabSynth (off / k=2 / k=4) and the prior (on / off) by nested grouped validation on the training
data only, and fits CatBoost. `selection_` shows every recipe tried. The estimators follow the sklearn contract
(`clone`, `get_params`, `cross_val_score` with `GroupKFold` and `tabagent.scorer("p@r")`).
