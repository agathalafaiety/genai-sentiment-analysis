# Dados

O projeto usa o **B2W-Reviews01**, corpus de mais de 130 mil avaliações de e-commerce em português coletadas em 2018. A fonte oficial arquivada está em <https://github.com/americanas-tech/b2w-reviews01>.

## Licença e atribuição

O corpus é distribuído sob **CC BY-NC-SA 4.0**. Seu uso é não comercial, requer atribuição à B2W Digital e derivados de dados devem manter licença compatível. A licença MIT deste repositório vale somente para o código original, não para o dataset.

Os arquivos brutos e processados não são versionados. O script fixa a revisão de origem `4639429ec698d7821fc99a0bc665fa213d9fcd5a` e baixa o CSV programaticamente.

## Rótulos

- 1–2 estrelas → `negative`
- 3 estrelas → `neutral`
- 4–5 estrelas → `positive`

Esse mapeamento é um proxy: a nota pode refletir entrega, preço ou atendimento e pode discordar do texto. Reviews vazias e duplicatas textuais são removidas antes do split.

## Modos

- `quick`: até 1.200 exemplos por classe, balanceados e determinísticos.
- `full`: todos os textos válidos, preservando o desbalanceamento natural.

Em ambos, a divisão estratificada é 64% treino, 16% validação e 20% teste com seed 42.
