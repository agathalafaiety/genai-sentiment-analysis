"""Gera notebooks pequenos, portaveis e sem estado oculto."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha256(f"markdown:{text}".encode()).hexdigest()[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha256(f"code:{source}".encode()).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON = """from pathlib import Path
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
"""


def write(name: str, cells: list[dict]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / name).write_text(
        json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> None:
    write(
        "01_eda.ipynb",
        [
            markdown(
                "# 01 — Análise exploratória\n\nObjetivo: verificar qualidade, distribuição de classes e comprimento dos textos no modo `quick`."
            ),
            code(COMMON),
            code("""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentiment_analysis.data import load_splits

splits = load_splits("quick")
data = pd.concat(splits.values(), keys=splits.keys(), names=["split"])
data.groupby(["split", "label"]).size().unstack(fill_value=0)
"""),
            code("""data = data.reset_index(drop=True)
data["characters"] = data["text"].str.len()
display(data.groupby("label")["characters"].describe().round(1))
sns.countplot(data=data, x="label", order=["negative", "neutral", "positive"])
plt.title("Distribuição das classes — amostra quick")
plt.show()
"""),
            markdown(
                "## Conclusão\n\nA amostra rápida é balanceada por construção. A nota 3 é tratada como neutra; isso é um *proxy* e pode discordar do sentimento textual."
            ),
        ],
    )
    write(
        "02_ml_baseline.ipynb",
        [
            markdown(
                "# 02 — Baseline e ML clássico\n\nObjetivo: comparar classe majoritária, Regressão Logística e Complement Naive Bayes sem vazamento de dados."
            ),
            code(COMMON),
            code("""import json
from sentiment_analysis.config import METRICS_DIR

metrics = json.loads((METRICS_DIR / "quick_classical_metrics.json").read_text(encoding="utf-8"))
[(name, round(values["macro_f1"], 4)) for name, values in metrics["results"].items()]
"""),
            code("""import pandas as pd
comparison = pd.read_csv(METRICS_DIR / "quick_comparison.csv")
comparison.sort_values("macro_f1", ascending=False)
"""),
            markdown(
                "## Conclusão\n\nO melhor pipeline é escolhido somente pelo Macro F1 de validação. As métricas acima vêm do conjunto de teste mantido fora da seleção."
            ),
        ],
    )
    write(
        "03_transformer.ipynb",
        [
            markdown(
                "# 03 — Transformer\n\nObjetivo: avaliar uma revisão fixada de um DistilBERT multilíngue de três classes no teste B2W. A inferência é opcional porque baixa cerca de 540 MB."
            ),
            code(COMMON),
            code("""import json
from sentiment_analysis.config import METRICS_DIR, TRANSFORMER_MODEL_ID, TRANSFORMER_REVISION

path = METRICS_DIR / "quick_transformer_metrics.json"
if path.exists():
    transformer_metrics = json.loads(path.read_text(encoding="utf-8"))
    display({key: transformer_metrics[key] for key in ("macro_f1", "weighted_f1", "latency_ms_per_sample", "n_samples")})
else:
    print("N/A: execute `python -m sentiment_analysis.transformer --mode quick --device -1`.")
print(TRANSFORMER_MODEL_ID, TRANSFORMER_REVISION)
"""),
            markdown(
                "## Conclusão\n\nA comparação deve considerar mudança de domínio: o Transformer foi destilado em sentimentos multilíngues, não ajustado nas reviews B2W."
            ),
        ],
    )
    write(
        "04_genai_analysis.ipynb",
        [
            markdown(
                "# 04 — IA generativa\n\nObjetivo: inspecionar prompts zero/few-shot e validar o contrato JSON sem efetuar chamadas pagas."
            ),
            code(COMMON),
            code("""from sentiment_analysis.genai import load_prompt, parse_genai_response

print(load_prompt("zero_shot_v1.txt"))
example = parse_genai_response('{"sentiment":"positive","confidence":0.91,"explanation":"Há elogio direto ao produto."}')
example.model_dump()
"""),
            code("""print(load_prompt("few_shot_v1.txt"))
print("Métricas GenAI: N/A enquanto nenhum provedor real for executado; mocks não são resultado científico.")
"""),
            markdown(
                "## Conclusão\n\nO fluxo local não requer chave. Uma avaliação real pode ser ativada explicitamente; cobertura e falhas são registradas junto às métricas."
            ),
        ],
    )
    write(
        "05_model_comparison.ipynb",
        [
            markdown(
                "# 05 — Comparação e análise de erros\n\nObjetivo: consolidar abordagens e investigar divergências no mesmo teste."
            ),
            code(COMMON),
            code("""import pandas as pd
comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
comparison
"""),
            code("""from sentiment_analysis.config import METRICS_DIR
predictions = pd.read_csv(METRICS_DIR / "quick_test_predictions.csv")
model_columns = [column for column in predictions if column.startswith("prediction_")]
errors = predictions[(predictions[model_columns] != predictions["label"].to_numpy()[:, None]).any(axis=1)]
display(errors[["text", "label", *model_columns]].head(12))
"""),
            markdown(
                "## Conclusão\n\nLeia os erros como hipóteses: ironia, negação, ortografia informal, avaliações mistas e o uso de estrelas como rótulo criam casos inerentemente difíceis."
            ),
        ],
    )
    write(
        "demo_colab.ipynb",
        [
            markdown(
                "# Demo no Google Colab\n\nTreina gratuitamente o pipeline clássico em uma amostra real e permite testar uma frase. Em Colab, ative uma GPU apenas para as etapas opcionais de Transformer/GenAI local."
            ),
            code("""import os, pathlib
if pathlib.Path.cwd().name != "genai-sentiment-analysis":
    !git clone https://github.com/agathalafaiety/genai-sentiment-analysis.git
    os.chdir("genai-sentiment-analysis")
!pip -q install -e .
"""),
            code("""from sentiment_analysis.data import prepare_dataset
from sentiment_analysis.training import train_classical_models
import pandas as pd

paths = prepare_dataset("quick")
splits = {name: pd.read_csv(path) for name, path in paths.items()}
metadata = train_classical_models(splits, mode="quick")
print("Melhor modelo:", metadata["best_model"])
"""),
            code("""from sentiment_analysis.inference import ClassicalPredictor

texto = "O produto chegou rápido e superou minhas expectativas."
previsao = ClassicalPredictor(mode="quick").predict(texto)
previsao.model_dump()
"""),
            markdown(
                "## GenAI opcional\n\nChaves nunca devem ser escritas no notebook. Use `google.colab.userdata.get` para ler um secret e execute `scripts/evaluate_genai.py` somente se quiser autorizar chamadas. Para um modelo aberto, instale `requirements-transformer.txt` e use `--provider local`."
            ),
        ],
    )


if __name__ == "__main__":
    main()
