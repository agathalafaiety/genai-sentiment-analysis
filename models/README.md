# Modelos

Modelos binários não são versionados. Gere o melhor pipeline clássico com:

```bash
python -m sentiment_analysis.training --mode quick --prepare
```

O arquivo `best_classical_quick.joblib` é escolhido pelo **Macro F1 de validação**, nunca pelo teste. O teste é usado uma única vez para a estimativa final.

O Transformer opcional é `lxyuan/distilbert-base-multilingual-cased-sentiments-student`, revisão `cf991100d706c13c0a080c097134c05b7f436c45`, licença Apache 2.0. Os pesos permanecem no cache local do Hugging Face.
