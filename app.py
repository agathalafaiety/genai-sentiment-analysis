"""Demonstração Streamlit do melhor modelo clássico treinado."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_analysis.config import MODELS_DIR
from sentiment_analysis.training import ClassicalPredictor, Prediction

MODEL_PATH = MODELS_DIR / "best_classical_quick.joblib"
METADATA_PATH = MODELS_DIR / "best_classical_quick.metadata.json"
EXAMPLES = {
    "Escolha um exemplo": "",
    "Positivo": "Adorei o produto, chegou antes do prazo e funciona perfeitamente!",
    "Neutro": "Produto entregue conforme descrito.",
    "Negativo": "Péssima compra: veio quebrado e o atendimento não resolveu nada.",
    "Ambíguo": "A câmera é ótima, mas a bateria deixa bastante a desejar.",
}
SENTIMENTS = {
    "positive": ("Positivo", "✅"),
    "neutral": ("Neutro", "●"),
    "negative": ("Negativo", "⚠️"),
}


@st.cache_resource(show_spinner=False)
def load_predictor(path: str, modified_at: float) -> ClassicalPredictor:
    """Mantém o modelo em memória e invalida o cache quando o arquivo muda."""
    del modified_at
    return ClassicalPredictor(Path(path))


def model_macro_f1(path: Path = METADATA_PATH) -> float | None:
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    value = metadata.get("best_model_test_metrics", {}).get("macro_f1")
    return float(value) if value is not None else None


def render_prediction(prediction: Prediction) -> None:
    name, icon = SENTIMENTS[prediction.sentiment]
    message = f"{icon} Sentimento **{name}** com **{prediction.confidence:.1%}** de confiança."
    if prediction.sentiment == "positive":
        st.success(message)
    elif prediction.sentiment == "negative":
        st.error(message)
    else:
        st.info(message)

    confidence, latency = st.columns(2)
    confidence.metric("Confiança", f"{prediction.confidence:.1%}")
    latency.metric("Latência", f"{prediction.latency_ms:.2f} ms")

    chart = pd.DataFrame(
        {
            "Sentimento": [SENTIMENTS[label][0] for label in prediction.probabilities],
            "Probabilidade": list(prediction.probabilities.values()),
        }
    ).set_index("Sentimento")
    st.subheader("Distribuição das probabilidades")
    st.bar_chart(chart, y="Probabilidade", horizontal=True)


def main() -> None:
    st.set_page_config(
        page_title="Análise de Sentimentos",
        page_icon="💬",
        layout="centered",
    )

    with st.sidebar:
        st.header("Sobre o modelo")
        macro_f1 = model_macro_f1()
        st.metric("Macro F1", f"{macro_f1:.4f}".replace(".", ",") if macro_f1 else "—")
        st.write("TF-IDF + Regressão Logística treinado no B2W-Reviews01.")
        st.caption("Execução local, sem envio do texto para APIs externas.")
        st.link_button(
            "Ver código no GitHub",
            "https://github.com/agathalafaiety/genai-sentiment-analysis",
            use_container_width=True,
        )

    st.title("Análise de Sentimentos")
    st.caption("Classifique avaliações de produtos como positivas, neutras ou negativas.")

    if not MODEL_PATH.exists():
        st.error("O modelo ainda não está disponível.")
        st.code("python -m sentiment_analysis.training --mode quick --prepare")
        st.stop()

    with st.container(border=True):
        example = st.selectbox("Exemplo", list(EXAMPLES))
        text = st.text_area(
            "Avaliação",
            value=EXAMPLES[example],
            height=150,
            placeholder="Digite ou cole uma avaliação…",
            max_chars=5_000,
        )
        analyze = st.button("Analisar sentimento", type="primary", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Digite um texto antes de analisar.")
        else:
            try:
                predictor = load_predictor(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                render_prediction(predictor.predict(text))
            except (OSError, ValueError) as exc:
                st.error(f"Não foi possível executar a análise: {exc}")

    with st.expander("Como interpretar o resultado"):
        st.write(
            "A confiança é uma estimativa do classificador, não uma garantia de acerto. "
            "Ironia, frases mistas e textos fora do domínio de e-commerce são mais difíceis."
        )

    st.caption("Projeto educacional — não use a previsão isoladamente para decidir sobre pessoas.")


if __name__ == "__main__":
    main()
