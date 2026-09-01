# GenAI Sentiment Analysis — português

[![CI](https://github.com/agathalafaiety/genai-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/agathalafaiety/genai-sentiment-analysis/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agathalafaiety/genai-sentiment-analysis/blob/main/notebooks/demo_colab.ipynb)

Projeto enxuto e reproduzível de análise de sentimentos em português. Compara baseline, Machine Learning clássico, Transformer e uma integração GenAI opcional, sem exigir API paga no fluxo principal.

![Demo Streamlit](reports/figures/streamlit_demo.png)

## Resultados reais

Avaliação no modo `quick`, com 720 reviews de teste balanceadas e execução em CPU:

| Modelo | Macro F1 | Latência média |
|---|---:|---:|
| Baseline majoritário | 0,1667 | 0,0006 ms/item |
| **TF-IDF + Regressão Logística** | **0,7504** | **0,0568 ms/item** |
| TF-IDF + Complement Naive Bayes | 0,7262 | 0,0701 ms/item |
| Transformer zero-shot | 0,4879 | 51,0110 ms/item |
| GenAI zero/few-shot | N/A | N/A |

| Melhor modelo clássico | Transformer zero-shot |
|---|---|
| ![Matriz de confusão da Regressão Logística](reports/figures/quick_logistic_regression_confusion_matrix.png) | ![Matriz de confusão do Transformer](reports/figures/quick_transformer_confusion_matrix.png) |

A Regressão Logística foi o melhor modelo. O Transformer raramente reconheceu a classe neutra, mostrando que um modelo maior não substitui alinhamento de domínio.

Os resultados completos estão em [`reports/model_comparison.csv`](reports/model_comparison.csv) e no notebook [`analysis.ipynb`](notebooks/analysis.ipynb).

## Dados e metodologia

O projeto usa o [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), corpus público de reviews brasileiras de e-commerce sob licença **CC BY-NC-SA 4.0**. A licença MIT deste repositório cobre apenas o código.

Mapeamento dos rótulos:

- 1–2 estrelas: `negative`
- 3 estrelas: `neutral`
- 4–5 estrelas: `positive`

Após limpeza e deduplicação, restaram 129.331 reviews. O modo `quick` usa até 1.200 exemplos por classe; o modo `full` usa todos os textos válidos. O split é estratificado em treino, validação e teste com seed 42. O vocabulário TF-IDF é ajustado apenas no treino para evitar leakage.

O Transformer avaliado é [`lxyuan/distilbert-base-multilingual-cased-sentiments-student`](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student), revisão fixada e licença Apache 2.0.

## Executar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m sentiment_analysis.training --mode quick --prepare
streamlit run app.py
```

macOS/Linux usa `source .venv/bin/activate` no lugar da ativação PowerShell.

Outros comandos:

```bash
# Preparar dados
python -m sentiment_analysis.data --mode quick
python -m sentiment_analysis.data --mode full

# Transformer opcional
pip install -e .[transformer]
python -m sentiment_analysis.transformer --mode quick --device -1

# Testes e qualidade
pip install -e .[dev]
pytest
ruff check .
ruff format --check .
```

## GenAI opcional

Os prompts zero-shot e few-shot estão versionados em [`prompts/prompts.json`](prompts/prompts.json). A saída é validada no formato:

```json
{
  "sentiment": "positive",
  "confidence": 0.91,
  "explanation": "O texto contém um elogio direto."
}
```

Nenhuma chamada paga ocorre por padrão. O adaptador OpenAI usa Structured Outputs e lê `OPENAI_API_KEY` apenas do ambiente. Também existe um provedor local opcional baseado em Qwen2.5-1.5B-Instruct. Mocks são usados somente nos testes e não entram nas métricas científicas.

## Limitações

- As estrelas são rótulos fracos e podem discordar do texto.
- O corpus cobre e-commerce brasileiro de 2018.
- Ironia, avaliações mistas, negação e textos curtos continuam difíceis.
- O modo quick é artificialmente balanceado.
- O projeto não deve ser usado sozinho para decisões sobre pessoas.

## Estrutura

```text
├── app.py                    # demo Streamlit
├── notebooks/               # análise consolidada e Colab
├── prompts/prompts.json      # prompts GenAI versionados
├── reports/                 # comparação e figuras principais
├── src/sentiment_analysis/  # dados, modelos e inferência
└── tests/                   # testes essenciais
```

Código sob licença [MIT](LICENSE). Dataset B2W-Reviews01 sob CC BY-NC-SA 4.0. Autoria: [agathalafaiety](https://github.com/agathalafaiety).
