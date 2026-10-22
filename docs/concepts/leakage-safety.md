---
title: "Leakage safety"
description: "Every data-dependent stage is fitted inside the training portion of each fold: how TabAgent enforces it by construction, what it tests, and the evaluation protocol."
lead: "Why a cross-validated number from TabAgent can be trusted, and the one place it cannot without extra work."
kind: Concept
---

# Leakage safety

The paper requires every data-dependent stage to be fitted inside the training portion of each fold (Sec. 4.3),
the TabSchema loop included. In TabAgent the fitted stages are in-fold by construction; TabSchema discovery is
in-fold when you run it through `cross_validate_discovery`:

* `cross_validate` is the only code that decides what is training data; for every fold and seed it clones the
  model, fits it on the training groups and scores the held-out groups through a label-free view.
* Dependency statistics (candidate priors, co-usage rates, embedding centering, k-NN outcomes) are `fitted`
  card features, fitted on real training rows only; each records the groups it saw.
* TabSynth runs only inside `fit`, on training rows.
* TabSchema's extraction code sees only the trajectory *prefix* before the decision, with outcome fields removed.
* A tripwire refuses a feature that reproduces the label.
* TabSchema discovery is not a fitted card feature: a card discovered once on all trajectories, as `onboard`
  does, has seen the held-out tasks of any later cross-validation. `onboard`'s report therefore estimates the
  head on a fixed card, and `cross_validate_discovery` estimates the whole pipeline: it runs discovery on each
  training fold's trajectories only and applies that fold's extractor to its held-out decisions. The bundled
  cards of the public datasets are fixed in advance (frozen or hand-authored, see
  [Reproducing the paper](../reproducing.md)), so only their fitted statistics are refit per fold.

The test suite proves these properties (scrambling held-out labels does not change held-out scores; scoring is
not transductive; fitted statistics only see training groups; each fold's discovery sees only training tasks).

## The protocol

5-fold task-level cross-validation (the paper's folds are bundled with each public dataset), three seeds
(20260720, 20260726, 20260727), macro average over tasks (over steps for Mind2Web element selection), 95%
bootstrap intervals, and paired task-level bootstraps with 2,000 resamples for comparisons (an improvement is
claimed only when the interval excludes 0; paper App. M).
