"""The Colab quickstart (notebooks/quickstart.ipynb): valid, committed without outputs, offline cells run.

Cells are tagged ``install`` (the pip cell, for Colab), ``offline`` (no network) or ``network`` (downloads the
SkillsBench release and the e5 encoder). Only the ``offline`` cells run here, in a fresh kernel.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "quickstart.ipynb"
TAGS = {"install", "offline", "network"}


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NOTEBOOK, as_version=4)


def _tags(cell) -> set[str]:
    return set(cell.metadata.get("tags", []))


def test_notebook_is_valid_and_committed_without_outputs(nb):
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not cell.outputs, f"cell {cell.id}: clear the outputs before committing"
            assert cell.execution_count is None, f"cell {cell.id}: clear the execution count"


def test_every_code_cell_declares_what_it_needs(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert len(_tags(cell) & TAGS) == 1, f"cell {cell.id}: tag it one of {sorted(TAGS)}"
    assert any("offline" in _tags(c) for c in nb.cells)


def test_offline_cells_run(nb):
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    offline = nbformat.v4.new_notebook(
        cells=[c for c in nb.cells if c.cell_type == "code" and "offline" in _tags(c)],
        metadata=nb.metadata,
    )
    nbclient.NotebookClient(offline, timeout=600, kernel_name="python3").execute()  # raises on a cell error
