import json
from pathlib import Path

import nbformat


def test_notebooks_are_valid_and_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "01_eda.ipynb",
        "02_ml_baseline.ipynb",
        "03_transformer.ipynb",
        "04_genai_analysis.ipynb",
        "05_model_comparison.ipynb",
        "demo_colab.ipynb",
    }
    found = {path.name for path in (root / "notebooks").glob("*.ipynb")}
    assert found == expected
    for name in expected:
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert "Objetivo" in "".join(notebook["cells"][0]["source"]) or name == "demo_colab.ipynb"
        assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
        nbformat.validate(nbformat.from_dict(notebook))
