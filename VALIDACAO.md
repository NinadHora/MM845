# Validação desta entrega

Data: 8 de setembro de 2026. Ambiente Python 3.13.5, CPU.

**Foram processadas 46 células de código em oito cadernos, sem exceções não tratadas. Doze testes independentes passaram. Isso NÃO significa que toda comparação foi executada: o bloco de UMAP foi explicitamente pulado por ausência da dependência.**

Os tutoriais 01–07 e as partes de kernels/t-SNE/PCA do Tutorial 08 possuem saídas reais salvas. O Tutorial 08, exercício 3, contém o código da comparação com o pacote oficial UMAP, mas não seus resultados. A instalação foi tentada e a conexão externa não estava disponível; `estado_umap.json` conserva essa informação.

`VALIDACAO.json` contém contagem de células, tempos e estado por caderno. `TESTES.txt` contém os testes rápidos. `AMBIENTE_EXECUTADO.json` contém as versões efetivamente importadas. `requirements-base.txt` reproduz a pilha utilizada; `requirements.txt` adiciona as dependências de UMAP ainda não validadas.

A validação inclui normas de esferas, distâncias arco/corda, ortogonalidade de harmônicos, dimensão de polinômios restritos, área polar, discriminante, persistência H0 por MST, simetrias de CNNs e modelos de conjuntos, máscaras e reversão de área. Os próprios notebooks incluem outros `assert` associados a exercícios.

Os números são resultados das configurações, sementes e orçamentos presentes no código. Não são garantias universais, prova de correção matemática de todo o texto ou execução no computador pessoal de Nina. Os notebooks originais foram preservados como referência; **não se declara que todos os exemplos originais foram reexecutados sem alterações**.

## Reexecutar

```bash
python -m unittest discover -s tests -v
python executar.py --kernel mm845
# Somente um caderno, após instalar a dependência de UMAP:
python executar.py --kernel mm845 8
```

Sementes fixas ajudam na reprodução; versões, plataforma e aritmética de ponto flutuante podem alterar os valores finais. Resultados NaN explicitamente documentados indicam regimes singulares, tempos não alcançados ou estimativas de transição ausentes, não valores substituídos por zero.
