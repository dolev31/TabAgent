import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import tabagent
from tabagent import citation
from tabagent.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cite_formats():
    bib = tabagent.cite()
    assert bib == tabagent.__citation__ == citation.cite("bibtex")
    assert bib.startswith("@misc{levy2026tabagent,") and bib.endswith("}")
    assert bib.count("{") == bib.count("}")
    assert "Goldstein, Yinon" in bib and citation.DOI in bib
    apa = tabagent.cite("apa")
    assert apa.startswith("Levy, I., Shlomov, S.,") and "& Choshen, L. (2026)" in apa
    assert tabagent.cite("text") == f"{citation.TITLE} (Levy et al., 2026), arXiv:2602.16429"
    with pytest.raises(ValueError, match="format must be one of"):
        tabagent.cite("mla")


def test_bibtex_matches_every_page_that_prints_it():
    # the README, AGENTS.md and docs/project/citation.md print the same entry, aligned by hand: compare fields
    def fields(text):
        entry = text.split("```bibtex", 1)[1].split("```", 1)[0]
        pairs = re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", entry, re.M | re.S)
        return {k: " ".join(v.split()) for k, v in pairs}

    ours = fields(f"```bibtex\n{tabagent.cite()}\n```")
    assert len(ours) == 8 and ours["author"].startswith("Levy, Ido and")  # the parser found every field
    for page in ("README.md", "AGENTS.md", "docs/project/citation.md"):
        if (ROOT / page).exists():  # the sdist ships no docs/
            assert fields((ROOT / page).read_text()) == ours, page


def test_citation_cff_has_the_same_paper():
    cff = (ROOT / "CITATION.cff").read_text()
    preferred = cff.split("preferred-citation:", 1)[1]
    assert f'title: "{citation.TITLE}"' in preferred and f'doi: "{citation.DOI}"' in preferred
    assert re.findall(r"family-names: (\S+)", preferred) == [last for _, last in citation.AUTHORS]
    assert re.findall(r"given-names: (\S+)", preferred) == [first for first, _ in citation.AUTHORS]
    assert f"year: {citation.YEAR}" in preferred


def test_cli_cite(capsys):
    assert main(["cite"]) == 0
    assert capsys.readouterr().out.strip() == tabagent.cite()
    assert main(["cite", "--format", "apa"]) == 0
    assert capsys.readouterr().out.strip() == tabagent.cite("apa")


def test_import_prints_nothing():
    cmd = [sys.executable, "-c", "import tabagent"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out.stdout == "" and out.stderr == ""


def test_record_is_json_ready():
    r = citation.record()
    assert json.loads(json.dumps(r)) == r and r["doi"] == citation.DOI and r["bibtex"] == tabagent.cite()


def test_zenodo_record_has_the_same_authors_and_links_the_paper():
    path = ROOT / ".zenodo.json"
    if not path.exists():  # the sdist ships no .zenodo.json
        pytest.skip("no .zenodo.json")
    z = json.loads(path.read_text())
    assert [c["name"] for c in z["creators"]] == [f"{last}, {first}" for first, last in citation.AUTHORS]
    assert z["upload_type"] == "software" and z["license"] == "Apache-2.0"
    paper = z["related_identifiers"][0]
    assert paper["identifier"] == citation.DOI and paper["relation"] == "isSupplementTo"
