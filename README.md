# MM845 — Resoluções comentadas

**Tutoriais 01–08 · código, explicações em português e mini-relatórios.**

Este pacote substitui a estrutura de estudos vazia enviada anteriormente. Agora há respostas aos **32 exercícios numerados**, implementações, demonstrações quando cabíveis, resultados salvos e um relatório de cinco campos para cada tutorial. O roteiro do Tutorial 01 também está desenvolvido.

> **Ressalva de execução:** a comparação experimental com o pacote `umap-learn` no Tutorial 08, exercício 3, está escrita, mas não foi executada porque a dependência não pôde ser instalada neste ambiente. Nenhum resultado de UMAP foi inventado. As demais partes dos cadernos foram executadas; consulte [validação](VALIDACAO.md).

O arquivo original contém apenas os tutoriais 01–08. Não foram inventadas soluções dos tutoriais 09–13, que não estavam no ZIP recebido. Este repositório ainda não foi publicado na conta de Nina.

## Comece pelo caderno e pelo mini-relatório

| Tutorial | Tema | Exercícios | Arquivos | Execução |
|---|---|---:|---|---|
| 01 | Ambiente e reprodutibilidade | Roteiro | [Caderno](estudos/tutorial_01/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_01/README.md) | Roteiro testado |
| 02 | Nuvens de pontos em esferas | 12 | [Caderno](estudos/tutorial_02/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_02/README.md) | Executado |
| 03 | Problemas variacionais e modelos lineares | 6 | [Caderno](estudos/tutorial_03/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_03/README.md) | Executado |
| 04 | Redes neurais e espaços de funções | 3 | [Caderno](estudos/tutorial_04/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_04/README.md) | Executado |
| 05 | CNNs, simetria e filtros | 2 | [Caderno](estudos/tutorial_05/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_05/README.md) | Executado |
| 06 | Atenção, conjuntos e ordem | 3 | [Caderno](estudos/tutorial_06/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_06/README.md) | Executado |
| 07 | Agrupamento e geometria métrica | 3 | [Caderno](estudos/tutorial_07/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_07/README.md) | Executado |
| 08 | Kernels e redução de dimensão | 3 | [Caderno](estudos/tutorial_08/solucoes.ipynb) · [Mini-relatório](estudos/tutorial_08/README.md) | Executado, exceto UMAP |

Cada pasta `estudos/tutorial_XX/` contém `solucoes.ipynb` com saídas, `solucoes.py` com o mesmo código e explicações em comentários, `RESOLUCOES.md` para leitura do texto/código sem abrir Jupyter, `README.md` com o mini-relatório e `resultados/` com tabelas/figuras. Os exercícios não têm campos vazios para preenchimento. [Cobertura por exercício](COBERTURA.md).

As respostas distinguem resultados matemáticos de observações do treino e corrigem premissas excessivamente fortes quando necessário. Por exemplo, uma métrica melhor não garante um embedding melhor em toda execução; pooling pode quebrar uma simetria; e um modelo invariante não recupera o sinal de uma área que depende da ordem.

## Executar no seu computador

A entrega foi testada em Python 3.13, CPU. Execute os comandos a partir da pasta que contém este README:

```bash
python -m venv .venv
source .venv/bin/activate
# No PowerShell, use: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mm845 --display-name "Python (MM845)"
jupyter lab
```

Abra um `solucoes.ipynb` e selecione o kernel **Python (MM845)**. Para executar todos sem abrir a interface:

```bash
python executar.py --kernel mm845
# Só um tutorial:
python executar.py --kernel mm845 2
# Testes curtos, sem treinar todos os modelos:
python -m unittest discover -s tests -v
```

Sem UMAP, `requirements-base.txt` permite executar as partes já validadas; o notebook 08 registra a ausência do pacote e não apresenta a comparação como concluída. A reexecução atualiza os cadernos e resultados. Os tempos desta entrega constam em `VALIDACAO.json`, mas variam conforme o computador.

## Publicação e créditos

[Como publicar](PUBLICAR.md) · [Origem e referências](REFERENCIAS.md) · [Validação](VALIDACAO.md) · [Licença](LICENSE)

Material original de **Edward Hirst e Tomás S. R. Silva**, MM845/IMECC/Unicamp. Resoluções e código adicional preparados com assistência de IA como solicita a disciplina para estudo e revisão de Nina da Hora.`material_original/` conserva o material recebido e seus créditos.
