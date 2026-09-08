# Mini-relatório

**Objetivo:** medir o que embeddings preservam e distorcem.

**Ideia matemática:** dimensão intrínseca, dimensão linear e fidelidade global são perguntas distintas.

**Experimento:** Comparei kernels, 18 execuções de t-SNE no toro e 18 na esfera, com métricas fixas.

**Resultado:** No toro, o overlap intrínseco de t-SNE variou de 71.3% a 81.3%.

**Limite:** A comparação com UMAP está codificada, mas não executada sem a dependência.

[Resoluções e saídas](solucoes.ipynb) · [Código executável](solucoes.py)
