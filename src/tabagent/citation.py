"""How to cite TabAgent: ``tabagent.cite()`` and ``tabagent cite``.

One record of the paper feeds every place the citation appears: the BibTeX and APA strings below, the
``citation`` field of a saved head's ``head.json`` and of the ``GET /version`` answer of ``tabagent serve``,
and the one line ``tabagent onboard`` prints when a head is saved. ``CITATION.cff`` at the repository root
holds the same record for GitHub's "Cite this repository" button; a test keeps the two in step.

Nothing is printed at import time and nothing is sent anywhere.
"""

from __future__ import annotations

TITLE = "TabAgent: Distilling LLM Agent Components into Cheap Classifiers"
AUTHORS = (
    ("Ido", "Levy"),
    ("Segev", "Shlomov"),
    ("Avi", "Yaeli"),
    ("Nir", "Mashkif"),
    ("Asaf", "Adi"),
    ("Yinon", "Goldstein"),
    ("Eilam", "Shapira"),
    ("Leshem", "Choshen"),
)
YEAR = 2026
ARXIV_ID = "2602.16429"
DOI = f"10.48550/arXiv.{ARXIV_ID}"
URL = f"https://arxiv.org/abs/{ARXIV_ID}"
KEY = "levy2026tabagent"
CITATION_PAGE = "https://dolev31.github.io/TabAgent/project/citation/"

BIBTEX = f"""@misc{{{KEY},
  title  = {{{TITLE}}},
  author = {{{" and ".join(f"{last}, {first}" for first, last in AUTHORS)}}},
  year   = {{{YEAR}}},
  eprint = {{{ARXIV_ID}}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.CL}},
  doi    = {{{DOI}}},
  url    = {{{URL}}}
}}"""

_apa_names = [f"{last}, {first[0]}." for first, last in AUTHORS]
APA = (
    f"{', '.join(_apa_names[:-1])}, & {_apa_names[-1]} ({YEAR}). {TITLE}. arXiv preprint arXiv:{ARXIV_ID}. "
    f"https://doi.org/{DOI}"
)

FORMATS = ("bibtex", "apa", "text")


def cite(format: str = "bibtex") -> str:
    """The citation of the TabAgent paper.

    Parameters
    ----------
    format: ``"bibtex"`` (the default), ``"apa"``, or ``"text"`` (a one-line reference for a README or a
        model card).

    Examples
    --------
    ```python
    import tabagent

    print(tabagent.cite())  # BibTeX, ready to paste into a .bib file
    print(tabagent.cite("apa"))
    ```
    """
    if format == "bibtex":
        return BIBTEX
    if format == "apa":
        return APA
    if format == "text":
        return f"{TITLE} (Levy et al., {YEAR}), arXiv:{ARXIV_ID}"
    raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {format!r}")


def record() -> dict[str, str]:
    """The citation as a JSON-ready record: the ``citation`` field of ``head.json`` and ``GET /version``."""
    return {"text": cite("text"), "bibtex": BIBTEX, "doi": DOI, "url": URL}


def notice() -> str:
    """The one line ``tabagent onboard`` prints after it saves a head."""
    return "If TabAgent helps your work, please cite the paper: tabagent cite (BibTeX) or " + CITATION_PAGE
