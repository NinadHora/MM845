# Publicar no seu GitHub

O ZIP não foi publicado automaticamente e não contém credenciais. As soluções, saídas e relatórios estão em `estudos/`; o material dos professores está separado em `material_original/`.

## Repositório próprio, pelo terminal

Extraia o ZIP e abra o terminal na pasta `MM845-resolvido`. Crie no GitHub um repositório vazio com o nome escolhido, sem adicionar outro README. O endereço abaixo é um exemplo que precisa ser substituído pelo endereço real da sua conta/repositório.

```bash
git init
git add .
git commit -m "Adiciona resoluções comentadas dos tutoriais MM845 01–08"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
git push -u origin main
```

Use a autenticação normal do Git/GitHub no seu computador. Não escreva tokens dentro dos notebooks ou dos comandos que serão publicados. Se o Git solicitar nome/e-mail, configure sua identidade local antes do commit. Preserve a licença e os créditos dos professores.

## Repositório que já existe

Copie `estudos/`, `tests/` e os arquivos da raiz para uma branch do seu repositório. Preserve o README anterior antes de substituir a página inicial. Não execute `git init` nem troque o remote de um repositório existente sem conferir sua configuração. Revise `git status` e `git diff --stat`, faça um commit e envie essa branch.

## Estado que deve permanecer visível

A comparação experimental com `umap-learn` não foi executada nesta entrega. Depois de instalar as dependências completas e executar o Tutorial 08, revise a tabela gerada e atualize esse estado. Os textos identificam a assistência de IA e não atribuem a Nina uma execução que ainda não ocorreu no computador dela.
