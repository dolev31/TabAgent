---
title: "Citation"
description: "How to cite TabAgent: the paper TabAgent: Distilling LLM Agent Components into Cheap Classifiers, with BibTeX."
kind: "Project"
lead: "If you use TabAgent, please cite the paper."
---

# Citation

TabAgent is the code of the paper *TabAgent: Distilling LLM Agent Components into Cheap Classifiers*, by Ido Levy,
Segev Shlomov, Avi Yaeli, Nir Mashkif, Asaf Adi, Yinon Goldstein, Eilam Shapira and Leshem Choshen (IBM; Weizmann
Institute of Science), 2026. The preprint is on [arXiv:2602.16429](https://arxiv.org/abs/2602.16429). Its first
version carries the paper's earlier title, "TabAgent: A Framework for Replacing Agentic Generative Components with
Tabular-Textual Classifiers".

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

## From the package

The installed package prints the same entry, so it is at hand where the work is done:

```bash
tabagent cite                 # BibTeX
tabagent cite --format apa    # APA; --format text gives one line for a README or a model card
```

```python
import tabagent

print(tabagent.cite())  # also tabagent.__citation__
```

A head saved by `onboard` records the citation in its `head.json`, so it travels with every head you share,
and `tabagent serve` reports it at `GET /version`. `tabagent onboard` mentions it in one line when it saves a
head; nothing is printed at import time.

The repository also carries a [`CITATION.cff`](https://github.com/dolev31/TabAgent/blob/main/CITATION.cff), which
GitHub reads for its "Cite this repository" button, and an
[`AGENTS.md`](https://github.com/dolev31/TabAgent/blob/main/AGENTS.md) that tells coding agents how to cite
TabAgent in the projects they build with it.
