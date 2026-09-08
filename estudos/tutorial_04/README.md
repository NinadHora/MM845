# Mini-relatório

**Objetivo:** comparar capacidade de aproximação e treinamento das redes.

**Ideia matemática:** um alvo estar representável não garante que o otimizador o encontre.

**Experimento:** foram testados crossover, Tanh/ReLU, curvas amostrais, oito sementes e quatro frequências.

**Resultado:** a representação ReLU de |z| foi exata; o ensemble de largura 32 teve MSE 4.38e-05.

**Limite:** as comparações são condicionadas às arquiteturas e ao orçamento de épocas; tempos de modos não atingidos permanecem censurados.

[Resoluções e saídas](solucoes.ipynb) · [Código executável](solucoes.py)
