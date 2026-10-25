## What and why

<!-- What this changes and why, in two or three sentences a reviewer can check. Link the issue: "Closes #123". -->

## How it was tested

<!-- The test you added, and what it fails on without your change. For a result: the command and its output. -->

## AI assistance

<!-- Tick one. AI help is welcome; an honest answer is required (CONTRIBUTING.md#ai-assisted-contributions). -->

- [ ] None
- [ ] AI-assisted: I wrote the change, with AI suggestions or review
- [ ] AI-written: an AI tool or agent wrote most of it, and I have reviewed every line and can explain it

## Checklist

- [ ] `make pr-ready` passes (lint, tests, the strict docs build and the contribution gate)
- [ ] the change does one thing, and the issue it closes agreed on the approach
- [ ] new behavior has a test that fails without the change
- [ ] public functions, methods and classes I added or changed have docstrings and type annotations
- [ ] users will not notice the change, or `CHANGELOG.md` has a line for it under "Unreleased"
- [ ] no data, trajectories, weights, keys or personal information are added
- [ ] no paper number moves, or `experiments/paper_cells.yaml` is updated and the reason is given above
