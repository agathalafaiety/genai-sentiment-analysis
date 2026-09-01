# Contribuindo

Contribuições são bem-vindas para correções, novas análises ou melhorias de documentação.

1. Crie um ambiente virtual com Python 3.11–3.13.
2. Instale `requirements-dev.txt`.
3. Crie uma branch curta e focada.
4. Execute `ruff check .`, `ruff format --check .` e `pytest`.
5. Não envie datasets, pesos, caches, notebooks com secrets ou métricas inventadas.

Ao adicionar um experimento, registre dataset/revisão, seed, split, hardware observado e limitações. Resultados de mocks devem ficar claramente separados de resultados científicos.
