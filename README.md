<h1 align="center">GenAI Sentiment Analysis</h1>

<p align="center">
  Classificação de sentimentos em avaliações de e-commerce brasileiro: do baseline simples a ML clássico, Transformers e GenAI opcional.
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://scikit-learn.org/"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&amp;logoColor=white"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&amp;logoColor=white"></a>
  <a href="LICENSE"><img alt="Licença MIT" src="https://img.shields.io/badge/Licen%C3%A7a-MIT-2EA44F"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/agathalafaiety/genai-sentiment-analysis/blob/main/notebooks/demo_colab.ipynb">Abrir no Colab</a> ·
  <a href="notebooks/analysis.ipynb">Ver análise</a> ·
  <a href="reports/model_comparison.csv">Explorar resultados</a>
</p>

## Visão geral

O projeto compara quatro abordagens no mesmo fluxo de avaliação reproduzível:

- Baseline de classe majoritária
- TF-IDF com Regressão Logística e Complement Naive Bayes
- Transformer multilíngue
- Provedores GenAI opcionais com zero-shot e few-shot

O fluxo principal roda localmente em CPU e não exige API paga.

## Resultados

Avaliação rápida com 720 avaliações balanceadas no conjunto de teste:

| Modelo | Macro F1 | Latência por item |
|---|---:|---:|
| **TF-IDF + Regressão Logística** | **0,7504** | **0,0568 ms** |
| TF-IDF + Complement Naive Bayes | 0,7262 | 0,0701 ms |
| Transformer multilíngue | 0,4879 | 51,0110 ms |
| Baseline majoritário | 0,1667 | 0,0006 ms |

A Regressão Logística apresentou o melhor equilíbrio entre qualidade, velocidade e simplicidade.

## Como executar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m sentiment_analysis.training --mode quick --prepare
streamlit run app.py
```

No macOS ou Linux, ative o ambiente com `source .venv/bin/activate`.

## Estrutura

```text
app.py                    Demonstração em Streamlit
notebooks/                Análise e notebook para Colab
prompts/prompts.json      Prompts GenAI versionados
reports/                  Resultados e figuras principais
src/sentiment_analysis/   Dados, treinamento, inferência e provedores
tests/                    Testes essenciais
```

## Recursos opcionais

```bash
# Transformer
pip install -e .[transformer]
python -m sentiment_analysis.transformer --mode quick --device -1

# Provedor OpenAI
pip install -e .[genai]

# Testes e qualidade
pip install -e .[dev]
pytest
ruff check .
```

As chamadas GenAI ficam desativadas por padrão. As credenciais são lidas apenas de variáveis de ambiente, e as respostas são validadas com um esquema Pydantic estrito.

## Dados

O projeto utiliza o conjunto público [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01). As notas são convertidas assim:

| Nota | Classe |
|---|---|
| 1–2 estrelas | `negative` |
| 3 estrelas | `neutral` |
| 4–5 estrelas | `positive` |

As divisões são estratificadas e determinísticas, com seed 42. O TF-IDF é ajustado apenas nos dados de treino para evitar vazamento de informação.

O código está sob a [licença MIT](LICENSE). O B2W-Reviews01 permanece sob sua licença original CC BY-NC-SA 4.0.
