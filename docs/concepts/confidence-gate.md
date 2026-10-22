---
title: "The confidence gate"
description: "How a TabAgent head defers unsure decisions to the original LLM: the confidence signal, calibrating the threshold with gate_coverage, and the service's deferral contract."
kind: "Concept"
lead: "A head that knows when it is unsure can step aside. The gate turns that into a threshold you set once."
---

# The confidence gate

The head replaces an LLM call, but it does not have to answer every decision. A **confidence-gated fallback** (paper
App. N.3) marks a decision whose confidence is below a threshold as *deferred*, and the agent then calls the
original LLM component for that decision. Deferral never makes the agent worse than before the head: it falls back
to the component the head replaced. It costs what the LLM costs, so the deferral rate is the number to watch.

The paper reports that TabHead's confidence flags its own errors at AUROC 0.88, which is why a gate can work, and
App. O.2 runs the gate in the loop.

## The confidence signal

For a ranking, `tabagent.tabhead.confidence` gives one value per decision, by one of two methods:

| Method | Definition | Use |
|---|---|---|
| `largest_gap` (default) | the largest drop between consecutive ranked scores | cutoff-free, so deployable |
| `margin_at_r` | score of the R-th candidate minus score of the (R+1)-th, where R is the size of the relevant set | needs R, an oracle quantity, so for analysis only |

For a single-class decision, the confidence is the probability of the predicted class.

## Use the gate

=== "Calibrate while onboarding"

    ```python
    head = tabagent.onboard("./logs", decision="ToolShortlist", gate_coverage=0.9)
    head.gate_threshold  # decisions below this confidence are deferred
    ```

    `gate_coverage=0.9` calibrates the threshold on the cross-validated decisions so that 90% stay with the head
    and the least confident 10% go back to the LLM. On the command line: `tabagent onboard ... --gate-coverage 0.9`.
    The gate is calibrated on ranking scores, so it does not apply to `kind="single"`.

=== "Calibrate an existing head"

    ```python
    head.calibrate_gate(cv_result, target_coverage=0.9)
    ```

=== "A threshold per call"

    ```python
    # a TabAgent (rows that carry the training columns)
    out = agent.decide(rows, k=5, defer_below=0.5, fallback=ask_the_llm)
    out[["decision", "candidate", "confidence", "deferred"]]
    out.attrs["fallback"]  # what `fallback` returned for the deferred decisions' rows

    # an onboarded head (a trajectory prefix)
    head.decide(prefix, candidates, k=5, defer_below=0.5)  # defaults to the calibrated gate_threshold
    ```

`fallback` is called with the rows of the deferred decisions, typically the original LLM component, and its return
value is stored in `result.attrs["fallback"]`.

## Choose a threshold

`tabagent.tabhead.risk_coverage(confidence_values, correct)` returns, for each coverage level (the fraction of
decisions kept), the mean correctness of the kept decisions. Plot it to see how much quality you buy by deferring
more, which is the paper's Fig. 14.

## Deferral is not only the gate

A decision is also deferred when the extractor fails on the prefix (a field the training prefixes had is missing,
the code raises, the sandbox cannot start), whatever its confidence, and, in the HTTP service, when the extraction
runs past its deadline. In every case the answer carries `"deferred": true`, and `error` says why unless the cause
is the gate. The agent's side of the contract is in the
[production runbook](../guides/production-runbook.md#timeouts-and-the-fallback-policy).

## When the gate misleads

* Calibration is a property of the training distribution. If the live decisions drift, the deferral rate moves
  with no change to the head: see [Drift, and when to re-onboard](../guides/production-runbook.md#drift-and-when-to-re-onboard).
* A deferral rate that rises with no drift means the threshold no longer fits: recalibrate with
  `onboard --gate-coverage` on recent logs.
