# Mini-relatório

**Objetivo:** medir o efeito de simetria, aumento de dados e filtros.

**Ideia matemática:** equivariância depende também do padding e do passo do pooling.

**Experimento:** Comparei MLP/CNN, quatro tamanhos de treino, translações, área e reflexões.

**Resultado:** No teste transladado, CNN=0.544 e MLP=0.553; a média D4 passou no teste de invariância.

**Limite:** Duas sementes e rasters finitos não estabelecem uma vantagem universal de amostragem.

[Resoluções e saídas](solucoes.ipynb) · [Código executável](solucoes.py)
