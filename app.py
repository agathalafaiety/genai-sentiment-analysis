"""Demonstracao Streamlit do melhor modelo classico treinado."""

from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_analysis.config import MODELS_DIR
from sentiment_analysis.training import ClassicalPredictor

st.set_page_config(
    page_title="Sentimento PT-BR",
    page_icon="💬",
    layout="centered",
)

EXAMPLES = {
    "Escolha um exemplo": "",
    "Positivo": "Adorei o produto, chegou antes do prazo e funciona perfeitamente!",
    "Neutro": "Produto entregue conforme descrito.",
    "Negativo": "Péssima compra: veio quebrado e o atendimento não resolveu nada.",
    "Ambíguo": "A câmera é ótima, mas a bateria deixa bastante a desejar.",
}


@st.cache_resource
def load_predictor(path: str) -> ClassicalPredictor:
    return ClassicalPredictor(Path(path))


st.title("Análise de sentimentos em português")
st.caption("Demonstração do pipeline TF-IDF selecionado por Macro F1 na validação.")

model_path = MODELS_DIR / "best_classical_quick.joblib"
if not model_path.exists():
    st.error("O modelo rápido ainda não foi treinado.")
    st.code("python -m sentiment_analysis.training --mode quick --prepare", language="powershell")
    st.stop()

model_label = "Melhor modelo clássico (quick)"
st.selectbox("Modelo disponível", [model_label], disabled=True)
example_name = st.selectbox("Exemplos em português", list(EXAMPLES))
text = st.text_area(
    "Texto para análise",
    value=EXAMPLES[example_name],
    height=150,
    placeholder="Digite ou cole uma avaliação em português…",
    max_chars=5_000,
)

if st.button("Analisar sentimento", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Digite um texto antes de analisar.")
    else:
        try:
            prediction = load_predictor(str(model_path)).predict(text)
        except Exception as exc:
            st.error(f"Não foi possível executar a inferência: {exc}")
        else:
            names = {
                "positive": "Positivo",
                "neutral": "Neutro",
                "negative": "Negativo",
            }
            st.success(
                f"Sentimento: **{names[prediction.sentiment]}** "
                f"— confiança {prediction.confidence:.1%}"
            )
            chart = pd.DataFrame(
                {
                    "sentimento": [names[label] for label in prediction.probabilities],
                    "probabilidade": list(prediction.probabilities.values()),
                }
            ).set_index("sentimento")
            st.bar_chart(chart, y="probabilidade", horizontal=True)
            st.caption(f"Latência desta inferência: {prediction.latency_ms:.2f} ms")

with st.expander("Como interpretar"):
    st.write(
        "A confiança é a probabilidade estimada pelo classificador clássico. "
        "Ela não garante que a previsão esteja correta, especialmente em ironia, "
        "sarcasmo, frases mistas ou textos fora do domínio de e-commerce."
    )

st.divider()
st.caption("Projeto educacional — não use a previsão como única base para decisões sobre pessoas.")
