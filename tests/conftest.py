from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def labeled_frame() -> pd.DataFrame:
    rows = []
    phrases = {
        "negative": ["produto horrível", "compra péssima", "não gostei", "veio quebrado"],
        "neutral": ["produto entregue", "chegou ontem", "contém duas peças", "pedido registrado"],
        "positive": ["produto excelente", "compra ótima", "adorei muito", "funciona perfeitamente"],
    }
    ratings = {"negative": 1, "neutral": 3, "positive": 5}
    for label, texts in phrases.items():
        for repeat in range(5):
            for text in texts:
                rows.append(
                    {"text": f"{text} exemplo {repeat}", "label": label, "rating": ratings[label]}
                )
    return pd.DataFrame(rows)
