<h1 align="center">GenAI Sentiment Analysis</h1>

<p align="center">
  Sentiment classification for Brazilian e-commerce reviews, from a simple baseline to classical ML, Transformers, and optional GenAI.
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://scikit-learn.org/"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&amp;logoColor=white"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&amp;logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-2EA44F"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/agathalafaiety/genai-sentiment-analysis/blob/main/notebooks/demo_colab.ipynb">Open in Colab</a> ·
  <a href="notebooks/analysis.ipynb">View analysis</a> ·
  <a href="reports/model_comparison.csv">Explore results</a>
</p>

## Overview

This project compares four approaches under the same reproducible evaluation pipeline:

- Majority-class baseline
- TF-IDF with Logistic Regression and Complement Naive Bayes
- Multilingual Transformer
- Optional zero-shot and few-shot GenAI providers

The default workflow is local, CPU-friendly, and does not require a paid API.

## Results

Quick-mode evaluation on 720 balanced test reviews:

| Model | Macro F1 | Latency / item |
|---|---:|---:|
| **TF-IDF + Logistic Regression** | **0.7504** | **0.0568 ms** |
| TF-IDF + Complement Naive Bayes | 0.7262 | 0.0701 ms |
| Multilingual Transformer | 0.4879 | 51.0110 ms |
| Majority baseline | 0.1667 | 0.0006 ms |

Logistic Regression achieved the best balance between quality, speed, and simplicity.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m sentiment_analysis.training --mode quick --prepare
streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Project structure

```text
app.py                    Streamlit demo
notebooks/                Analysis and Colab notebooks
prompts/prompts.json      Versioned GenAI prompts
reports/                  Results and key figures
src/sentiment_analysis/   Data, training, inference, and providers
tests/                    Essential test suite
```

## Optional models

```bash
# Transformer
pip install -e .[transformer]
python -m sentiment_analysis.transformer --mode quick --device -1

# OpenAI provider
pip install -e .[genai]

# Tests and linting
pip install -e .[dev]
pytest
ruff check .
```

GenAI calls are disabled by default. API credentials are read only from environment variables, and responses are validated with a strict Pydantic schema.

## Data

The project uses the public [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) dataset. Ratings are mapped as follows:

| Rating | Label |
|---|---|
| 1–2 stars | `negative` |
| 3 stars | `neutral` |
| 4–5 stars | `positive` |

Splits are stratified and deterministic with seed 42. TF-IDF is fitted only on training data to prevent leakage.

The repository code is licensed under [MIT](LICENSE). B2W-Reviews01 remains under its original CC BY-NC-SA 4.0 license.
