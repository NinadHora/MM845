# Origem e referências técnicas

## Material da disciplina

Edward Hirst e Tomás S. R. Silva. **MM845 — Tópicos de Geometria III: AI for Geometry**, IMECC/Unicamp, segundo semestre de 2026. Material recebido como `MM845-main(2).zip`, preservado em `material_original/`. O README original anuncia 13 tutoriais; o arquivo fornecido contém somente os tutoriais 01–08. A licença MIT original foi preservada.

Enunciados e títulos pertencem ao material da disciplina. As resoluções em português, implementações adicionais, testes e relatórios foram preparados com assistência de IA nesta conversa. Não são um gabarito oficial. A identidade e os créditos originais não foram substituídos pela identidade de Nina.

## Documentação primária consultada

- [SciPy: sph_harm_y](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html) — convenção de índices e ângulos dos harmônicos.
- [PyTorch 2.10: BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.BCEWithLogitsLoss.html) — logits e estabilidade numérica.
- [scikit-learn: TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) — matriz de distâncias pré-computada, perplexidade e `max_iter`.
- [UMAP: reprodutibilidade](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) — sementes e paralelismo.
- [umap-learn 0.5.9.post2](https://pypi.org/project/umap-learn/0.5.9.post2/) e [PyNNDescent 0.5.13](https://pypi.org/project/pynndescent/0.5.13/) — dependências especificadas para a comparação pendente.

A documentação pode exibir versões posteriores às usadas na execução. As versões efetivamente testadas estão em `AMBIENTE_EXECUTADO.json`; a dependência UMAP não foi executada nesta entrega.
