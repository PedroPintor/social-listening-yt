# Revisão do projeto contra os critérios do professor

Revisão de 15/11/2026, véspera da apresentação. Fontes: `docs/raw/passo_a_passo_do_projeto.docx`, o feedback da Parte 01, as notas, os READMEs, os notebooks 02 a 05c, os reports e o código. O código não foi alterado.

**Atualização (15/11):** as correções pequenas da seção 3 e os itens 4 a 6 da lista "Até a apresentação" foram aplicados no mesmo commit desta nota. Do item 3, entrou a leitura do `05` e do `05b`, mas não os exemplos de saída. Também entraram os números, o texto desatualizado, a nota sobre a B2W e a remoção dos avisos com caminho pessoal. O `05b` foi reexecutado. As tabelas abaixo mostram a situação **antes** das correções.

## 1. Situação por etapa e por critério

### Entregas (passo a passo)

| Etapa (data) | Peso | O que existe | Status | O que falta |
|---|---|---|---|---|
| 1. Proposta (22/10) | 10% | `docs/raw/Proposta — ….md` | ok, mas fora do git | `docs/raw/` está no `.gitignore`. A proposta entregue não está no repositório e ainda cita spaCy, B2W, pseudo-rotulagem, chatbot e o exemplo do brinquedo. |
| 2. Sentimentos (22/10) | 10% | `notebooks/02_…ipynb`: TF-IDF + RL, F1 0,883, erros, mistos e frases no estilo YouTube | ok | Título sem o número da etapa. O texto ainda cita a pseudo-rotulagem com a B2W. |
| 3. NER (29/10) | 10% | `notebooks/03_…ipynb`: GLiNER2, 24 frases sintéticas com 42 entidades, P/R/F1 por categoria e análise de erros | parcial | Não há avaliação por entidade em comentários **reais**. O notebook e o README prometem essa avaliação "com os rótulos do Pedro", mas a rotulagem não anotou entidades. |
| 4. Rede neural (29/10) | 15% | `notebooks/04_…ipynb`, `src/redes.py`, `reports/04_*.csv`. Fase 1 compara arquiteturas, Fase 2 compara embeddings; 3 sementes | ok | Nada essencial. |
| 5. Transformers (09/11) | 15% | `notebooks/05_…ipynb` (Colab), `05b`, `05c`, `reports/05*.csv` | parcial | A seção 7 do `05` ("Leitura dos resultados") está **"A preencher"**. O `05b` remete a um relatório que ainda não existe. Faltam exemplos de saída e uma análise qualitativa dos erros dos Transformers, pedidas na entrega. |
| 6. Protótipo + relatório + código (09/11) | 15% | `coleta_youtube/youtube.py` (cliente da API) e a decisão de escopo em `docs/notas/decisoes_de_escopo.md` | **falta** | O app Streamlit em `src/`, as instruções para executá-lo e o registro de falhas com entradas diferentes. O relatório técnico também falta. Pelo calendário, o prazo já passou. |
| 7. Apresentação (16/11) | 25% | nada no repositório | **falta** | Slides (`apresentacao.pdf`), roteiro da demonstração e divisão da fala entre Pedro e Maria Alice. |

### Avaliação detalhada (critérios)

| Critério | Peso | Evidência | Status | O que falta |
|---|---|---|---|---|
| Problema e qualidade dos dados | 10% | Proposta, RePro com licença citada no `02`, coleta estratificada com 50 itens por produto e no máximo 10 por vídeo, guia de rotulagem no `rotulador.html`, kappa (`limitacoes…md`), `youtube_api.md` | ok | Versionar a proposta, ou um resumo dela, fora de `docs/raw/`. |
| Sentimento e NER | 15% | `02`, `03`, `05c` (NER como filtro) | parcial | Avaliação de NER por entidade em dados reais, ou a justificativa de por que ela não foi feita. |
| Redes neurais e Transformers | 20% | `04`, `05`, `05b`, com a justificativa de cada escolha | ok, com lacunas de texto | Escrever a leitura do `05`, com custo, desempenho e limitações em texto, além da tabela. |
| Integração e funcionalidade | 20% | Só os componentes isolados e o `05c`, que simula o fluxo | **falta** | O app funcionando, de ponta a ponta. É o maior peso em aberto. |
| Avaliação e discussão | 15% | `05b`, `05c`: troca de domínio, ruído, bootstrap e McNemar, erro no resumo | ok | Consolidar a discussão no relatório. |
| Relatório e reprodutibilidade | 10% | `src-pln/README.md` cobre as etapas 2 a 4; `requirements.txt` | parcial | Relatório de 6 a 10 páginas, com a seção de aspectos éticos. Falta `README.md` na raiz, e o README atual não cobre a etapa 5, a coleta nem o app. No `requirements.txt` faltam `transformers`, `scipy` e `streamlit` fixados. |
| Apresentação | 10% | — | falta | Slides e ensaio, com respostas prontas para "por que não chatbot?" e "por que o NER não filtra?". |

## 2. Feedback do professor sobre a etapa 1

| # | Ponto | Respondido? | Onde / o que falta |
|---|---|---|---|
| 1 | Sobreposição RePro–B2W | Sim na prática, mas não registrado | A B2W não foi usada, então não há vazamento. Falta **dizer isso** (nota de escopo ou relatório). O `02` ainda cita a B2W como próximo passo. |
| 2 | Conversão de nota em sentimento | Sim na prática, mas não registrado | Ponto superado: só o RePro, rotulado por humanos, é usado. Os mistos são tratados no `02`, seção 2. Falta uma linha explicando. |
| 3 | NER sem base de avaliação; spaCy sem a classe produto | Parcial | O spaCy foi trocado pelo GLiNER2, com justificativa no README e no `03`. Há um gabarito sintético, e o catálogo da B2W foi abandonado. **Falta** o P/R/F1 por entidade em comentários reais do YouTube. O `05c` só avalia o NER como filtro. |
| 4 | Limitações da API | Sim | `docs/notas/youtube_api.md`, seções 2 a 4: período pela data do vídeo, cota e `madeForKids`. O brinquedo foi trocado por 6 produtos, com o histórico em `gerar_amostra.py`. |
| 5 | Amostra rotulada: balanceamento e guia | Sim | Amostra estratificada por produto, com limite por vídeo (`gerar_amostra.py`). Há um guia com regras para neutro e ironia (`rotulador.html`), 350 itens e kappa 0,48 / 0,63 (`limitacoes…md`). |
| 6 | Pseudo-rotulagem | Não registrado | Foi abandonada. Falta dizer por quê em uma linha (o RePro já tem rótulos humanos, e a B2W saiu). |
| 7 | Técnica do chatbot | Sim | `docs/notas/decisoes_de_escopo.md`: o app substitui o chatbot, com justificativa e a ideia de RAG como trabalho futuro. |

## 3. Inconsistências

**Números**
- **BiGRU, F1 no RePro:** 0,892 é a média das 3 sementes. O modelo salvo e avaliado no YouTube (`05b` e `05c`) é a semente 42, com F1 0,887. Mesmo assim, a coluna `f1_macro_repro` do `05b` e a tabela de `decisoes_de_escopo.md` mostram 0,892.
- **BiGRU, latência:** o README e o `04` dizem ~3,5 ms. Em `reports/04_resultados_rede_neural.csv`, a média é 8,9 ms, puxada pela semente 44 (19,5 ms), e é esse valor que o `05c` usa (2,7 s para 300 comentários). As notas dizem "~3 s". Escolher um número só.
- **Validação:** o `05`, seção 1, diz "mesma validação (10% do treino) das etapas 2 e 4". A etapa 4 usou 15% (5.429/959), a etapa 2 não teve validação e o `05` usou 10% (5.749/639).
- **Conclusão 4 do `05c`:** "TF-IDF erra 16 a 21 pp, as redes 6 a 10". Isso ignora o filtro por nome, em que o TF-IDF chega a 30,4 pp e as redes a 15,7–19,4 pp.
- **Bases diferentes nas notas:** `limitacoes…md` cita 174 comentários, e `decisoes_de_escopo.md` cita 86. As duas contas estão certas, mas medem coisas diferentes (todos os comentários × só os que são sobre o produto). Vale dizer qual é qual.

**Texto desatualizado**
- `limitacoes…md`, §4, "Uso no projeto": diz que o NER separa os comentários. A decisão atual é separar pela busca do nome e usar o NER para listar as marcas.
- Introdução do `05c`: "Filtro por NER … **É o fluxo do app**". Era a hipótese testada, não o fluxo final.
- `03`, seções 8 e 9, e README da etapa 3: prometem a "avaliação definitiva com comentários reais rotulados pelo Pedro". Ela não foi feita, e a rotulagem foi feita pelos dois.
- Introdução do `04`: "transforma as menções encontradas pelo NER". A conclusão do `04` ("a escolha final fica para depois da etapa 5") já foi resolvida.
- `src-pln/README.md`: termina em "Próximas etapas: Etapa 5 … Etapa 6: Chatbot".
- `requirements.txt`: o comentário "streamlit # aplicação / chatbot".
- `02`: o título é "# Etapa — Análise de sentimentos", sem o número.

**Células sem saída ou leitura vazia**
- Os notebooks foram todos executados em ordem e não têm erros. A única célula sem saída é a 5 do `05c` (dicionário `APELIDOS`), o que é esperado.
- O `05`, seção 7, está "A preencher". O `05b`, seção 4, só diz "ver o relatório".

**Caminhos pessoais em saídas versionadas**
- `03`, células 2 e 3, e `05c`, célula 3: avisos do tqdm e do torch (stderr) com o caminho do `.venv` na pasta do usuário. **Corrigido:** as saídas de stderr foram removidas.
- `03`, célula 2: a saída "Diretório atual: …" (stdout) ainda mostra a pasta do usuário. **Pendente.**
- O `05` mostra `/content/…` e a URL do GitHub, o que não é problema.

**Textos do YouTube**
- Nenhum trecho dos 318 textos da amostra aparece nos notebooks, nos reports, nas notas ou no README. A busca comparou os 25 primeiros caracteres de cada texto e não imprimiu nenhum deles.
- `coleta_youtube/rotulos/rotulos_consolidados.csv` versiona `video_id` e `comment_id`. Não há texto, mas esses IDs permitem recuperar o comentário e o autor. Isso é aceitável para a reprodutibilidade, mas deve ser citado em "Aspectos éticos". A alternativa é remover o `comment_id`.

## 4. Prioridades

**Até a apresentação (16/11)**
1. **O app Streamlit funcionando** (20% do critério de integração, mais 15% da entrega da etapa 6). Mínimo: produto e período, depois coleta, separação pelo nome, BERTimbau congelado, resumo e marcas pelo NER. Ter uma consulta já em cache para a demonstração não depender da cota nem da rede.
2. **Slides e roteiro** (25% + 10%): problema → dados e rotulagem (kappa) → escada de modelos (0,883 → 0,892 → 0,921 no RePro) → queda no YouTube → NER não filtra → escolha do BERTimbau congelado → demonstração → limitações. Dividir a fala entre os dois.
3. **Preencher a leitura do `05`, seção 7, e do `05b`, seção 4**: um parágrafo cada, com custo, desempenho e limitações. Incluir 3 ou 4 exemplos de saída com frases sintéticas.
4. **Corrigir os números divergentes**: BiGRU 0,892 × 0,887, latência 3,5 × 8,9 ms e o texto da validação no `05`.
5. **Ajustar o texto desatualizado**: `limitacoes` §4, introdução do `05c`, `03`/README da etapa 3 e título do `02`.
6. **Uma nota curta** respondendo aos itens 1, 2 e 6 do feedback (B2W e pseudo-rotulagem abandonadas e por quê). Pode entrar em `decisoes_de_escopo.md`.

**Para o relatório final**
- Relatório de 6 a 10 páginas com todas as seções, incluindo aspectos éticos: anonimização, IDs no CSV, licença CC BY-NC-SA do RePro, termos da API e vieses.
- `README.md` na raiz com integrantes, instalação, como rodar cada notebook e o app. Atualizar o `src-pln/README.md` com a etapa 5 e a coleta.
- `requirements.txt` completo: `transformers`, `scipy` e `streamlit` fixados.
- Avaliação de NER por entidade em ~50 comentários reais (item 3 do feedback), ou a justificativa explícita para não fazê-la.
- Registro de falhas do app com entradas diferentes (marca pequena, vídeos sem comentários, cota esgotada), como pede a etapa 6.
- Limpar as saídas com caminho pessoal do `03` e do `05c` (reexecutar ou suprimir os avisos).
- Versionar a proposta, ou um resumo atualizado dela, fora de `docs/raw/`.
