# Radar de Marca: monitoramento de marca no YouTube com PLN

Projeto da disciplina de Processamento de Linguagem Natural. O **Radar de Marca** responde a uma pergunta de quem cuida de uma marca: *o que o público está dizendo do meu produto nos vídeos recentes do YouTube?*

A pessoa digita o produto e um período. O app busca os vídeos publicados nesse período e lê os comentários. Depois, destaca os que citam o produto, classifica o sentimento de cada um (positivo, neutro ou negativo) e mostra quais outras marcas aparecem na conversa.

**Integrantes:** Pedro Garcia Pintor e Maria Alice.

## Etapas e resultados

| Etapa | O que foi feito | Onde |
|---|---|---|
| 1. Problema | Público, objetivo, fontes de dados e tecnologias | [`docs/proposta.md`](docs/proposta.md), com uma nota do que mudou depois |
| 2. Sentimento clássico | TF-IDF + Regressão Logística no RePro: F1 macro 0,883 | [`02_etapa_analise_sentimentos`](src-pln/notebooks/02_etapa_analise_sentimentos.ipynb) |
| 3. NER | GLiNER2 PT-BR com categorias livres (marca, produto, empresa, pessoa). Em 24 frases sintéticas: F1 0,60, ou 0,86 ignorando a categoria | [`03_reconhecimento_entidades`](src-pln/notebooks/03_reconhecimento_entidades.ipynb) |
| 4. Rede neural | BiLSTM, BiGRU e CNN; a melhor é a BiGRU com GloVe do NILC (F1 0,892) | [`04_rede_neural`](src-pln/notebooks/04_rede_neural.ipynb) |
| 5. Transformers | XLM-R de tweets, BERTimbau congelado + RL e BERTimbau ajustado (F1 0,921 no RePro) | [`05_transformers`](src-pln/notebooks/05_transformers.ipynb) (Colab) |
| 5b. Validação no YouTube | Todos os modelos em 350 textos reais rotulados por nós dois (kappa 0,48). O melhor é o BERTimbau congelado + RL, com F1 0,807 | [`05b_avaliacao_youtube`](src-pln/notebooks/05b_avaliacao_youtube.ipynb) |
| 5c. Fluxo do app | O NER não serve como filtro de relevância, e a busca pelo nome serve (Precision 0,91). Nenhum modelo difere do TF-IDF de forma significativa | [`05c_fluxo_do_app`](src-pln/notebooks/05c_fluxo_do_app.ipynb) |
| 6. Aplicação | App em Streamlit (sem chatbot; ver as decisões) | [`src/`](src/) |

O **relatório técnico** está em [`src-pln/reports/relatorio_tecnico.md`](src-pln/reports/relatorio_tecnico.md), e os slides da apresentação final, em [`apresentacao/`](apresentacao/). O detalhe de cada etapa está em [`src-pln/README.md`](src-pln/README.md). A decisão de fazer um painel em vez de chatbot está em [`docs/notas/decisoes_de_escopo.md`](docs/notas/decisoes_de_escopo.md). As limitações estão em [`docs/notas/limitacoes_e_trabalhos_futuros.md`](docs/notas/limitacoes_e_trabalhos_futuros.md).

## Estrutura

```
src/                     app (Streamlit)
  app.py                 interface: consulta, painel, vídeos, comentários, marcas citadas
  pipeline.py            coleta → separação pelo nome → sentimento → NER; salvar/abrir análises
  sentimento.py          BERTimbau congelado + Regressão Logística
  treinar_classificador.py  treina a Regressão Logística e salva em src-pln/models/
src-pln/
  notebooks/             etapas 2 a 5c
  src/                   código comum dos notebooks (dados.py, redes.py)
  coleta_youtube/        cliente da API do YouTube, amostra, rotulador e kappa (ver o README da pasta)
  data/RePro.csv         base de treino (10 mil avaliações da Americanas)
  reports/               resultados em CSV (sem textos do YouTube)
  utils/                 baixar o RePro e os embeddings
  models/                só o classificador do app vai para o git (bertimbau_congelado_rl.joblib)
docs/proposta.md         proposta da etapa 1
docs/notas/              decisões, limitações, API do YouTube, revisão dos critérios
modelos_baixados/        pesos do BERTimbau e do GLiNER2, criada na primeira análise (fora do git)
```

## Instalação

Usamos o Python 3.13 no Windows. Os comandos são executados a partir da raiz do projeto.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r src-pln/requirements.txt
```

## Chave da API do YouTube

O app busca os vídeos e os comentários pela YouTube Data API v3. Cada pessoa usa a própria chave, que é gratuita e não pede cartão:

1. Entre em <https://console.cloud.google.com/> com uma conta Google e crie um projeto.
2. Em **APIs e serviços → Biblioteca**, ative a **YouTube Data API v3**.
3. Em **APIs e serviços → Credenciais → Criar credenciais → Chave de API**, gere a chave.
4. Na raiz do projeto, copie `.env.example` para `.env` e cole a chave em `YOUTUBE_API_KEY=`.

O `.env` está no `.gitignore`: nunca suba a chave para o GitHub.

A cota gratuita é de 10.000 unidades por dia. Uma análise do app gasta cerca de 100 a 120 unidades, quase tudo na busca de vídeos, o que dá umas 80 análises por dia. Repetir a mesma consulta usa o cache e não gasta cota. Os custos de cada chamada e as limitações da API (região, vídeos infantis, busca não exaustiva) estão em [`docs/notas/youtube_api.md`](docs/notas/youtube_api.md).

## Como rodar o app

1. **Abrir o app:**

   ```bash
   python -m streamlit run src/app.py
   ```

   O app abre em `http://localhost:8501`.

2. **Analisar:** digite o produto e os apelidos separados por vírgula (por exemplo, `iphone, apple, ip17`). Depois, escolha o período e clique em **Analisar**. Em *Opções da coleta*, o app mostra quantas unidades da cota a consulta vai gastar, no máximo.

Com 5 vídeos e 60 comentários por vídeo, a análise leva cerca de 1 a 2 minutos na CPU.

Cada análise é salva em `src/analises/` e pode ser aberta de novo sem chave e sem gastar cota. Essa pasta fica fora do git, porque tem textos de comentários. Então, num clone novo, ela começa vazia.

### O classificador já vem pronto

O classificador de sentimento do app tem duas partes:

| Parte | Quem treinou | Tamanho | De onde vem |
|---|---|---|---|
| BERTimbau, usado **congelado** (sem mudar nenhum peso) | NeuralMind | ~440 MB | Baixado do Hugging Face na primeira análise |
| Regressão Logística em cima dos vetores do BERTimbau | Nós | 10 KB | Já está no repositório: `src-pln/models/bertimbau_congelado_rl.joblib` |

Então não é preciso treinar nada para usar o app. Sobre o `bertimbau_congelado_rl.joblib`:
- Foi gerado por `src/treinar_classificador.py`, com o treino do RePro, e tem F1 macro de 0,885 no teste do RePro.
- Foi salvo com o `scikit-learn==1.9.1`, a versão fixada no `requirements.txt`. Com outra versão, ele pode não abrir.
- Como foi treinado no RePro, segue a mesma licença, CC BY-NC-SA 4.0, ou seja, só para uso não comercial.
- Um arquivo joblib executa código quando é aberto, como qualquer *pickle* do Python. Só abra arquivos joblib de fonte confiável.

Para refazer o treino (opcional, alguns minutos na CPU), rode o comando abaixo. Ele sobrescreve o arquivo e mostra o F1 no teste.

```bash
python src/treinar_classificador.py
```

### Modelos baixados na primeira análise

Na primeira análise, o app baixa do Hugging Face os pesos de dois modelos públicos. Das próximas vezes, eles já estão no disco.

| Modelo | Para quê | Tamanho |
|---|---|---|
| [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (BERTimbau) | Sentimento | ~440 MB |
| [`ottema/gliner2-ptbr`](https://huggingface.co/ottema/gliner2-ptbr) (GLiNER2) | Marcas citadas (NER) | ~1,2 GB |

São **~1,6 GB** no total, e o download leva alguns minutos, conforme a internet. As bibliotecas que rodam esses modelos (`transformers` e `gliner2`) vêm do `requirements.txt`, mas os pesos não.

Os pesos ficam em **`modelos_baixados/`**, na raiz do projeto (fora do git), e não na pasta global do Hugging Face no seu usuário. Para remover os modelos, apague essa pasta. Para desinstalar tudo, apague a pasta do projeto: o `.venv` e os modelos vão juntos.

Os notebooks não usam essa pasta. Quando rodados localmente, baixam os modelos na pasta padrão do Hugging Face (`~/.cache/huggingface`).

## Como rodar os notebooks

Rode na ordem, com o kernel do `.venv`.

| Notebook | Precisa de | Observação |
|---|---|---|
| 02, 03 | `src-pln/data/RePro.csv` (já no repositório; se faltar: `python src-pln/utils/baixar_dados.py`) | CPU |
| 04 | Embeddings: `python src-pln/utils/baixar_embeddings.py` (~1,1 GB baixados; guarda só um recorte) | CPU basta; a GPU acelera |
| 05 | GPU: feito no **Google Colab** (as instruções estão no topo do notebook) | O ajuste fino do BERTimbau leva horas sem GPU |
| 05b, 05c | `src-pln/coleta_youtube/dados/amostra_rotulagem.csv` | Ver abaixo |

Os textos do YouTube não estão no repositório, porque os termos da API desaconselham redistribuir. Só os rótulos, sem texto, estão versionados, em `src-pln/coleta_youtube/rotulos/rotulos_consolidados.csv`. Para refazer o 05b e o 05c, é preciso coletar de novo com `src-pln/coleta_youtube/gerar_amostra.py`, seguindo o [README da coleta](src-pln/coleta_youtube/README.md).

Como os comentários do YouTube mudam com o tempo, uma coleta nova não traz exatamente os mesmos 350 textos. Os resultados que obtivemos estão em `src-pln/reports/05b_*.csv` e `05c_*.csv`.

Os testes do cliente da API não gastam cota:

```bash
python src-pln/coleta_youtube/teste_mock.py
```

## Registro de falhas do app

A etapa 6 pede o comportamento do app com entradas diferentes. As linhas marcadas como **reais** vêm de análises que rodamos com a API de verdade entre 13 e 14/11/2026. As outras foram verificadas com a API simulada (`mock_api.py`) ou pelo código.

| Entrada | O que aconteceu | Por quê | Como o app reage |
|---|---|---|---|
| **Produto grande** (real): iPhone 17, 08 a 14/11, apelidos `iphone, apple, ip17, ios` | Funcionou: 8 vídeos, 234 comentários, 69 (29%) citam o produto. Concorrentes: Samsung (21), S25 Ultra (19) | — | Painel completo. Mesmo assim: um vídeo de Portugal (`pt-PT`) passou no filtro de idioma, e o NER lista termos genéricos como "17" e "celular" |
| **Marca pequena** (real): Gelelé, 13 a 14/11 | 0 vídeos, ~100 unidades da cota gastas | Os poucos vídeos que a busca trouxe foram todos descartados: sem comentários (o YouTube desativa os comentários em vídeos infantis), marcados como infantis ou com áudio em outra língua | Mensagem "Nenhum comentário encontrado. Tente um período maior ou outro nome para o produto." |
| **Produto de nicho com nome em inglês** (real): Cube Squeeze Ball, 13 a 14/11 | 0 vídeos, ~100 unidades gastas | Nenhum vídeo aproveitável no período | A mesma mensagem |
| **Termo genérico** (real): slime | 3 vídeos, 105 comentários, só 12 (11%) citam o termo | Boa parte dos vídeos estava em outra língua e foi descartada. Com tão poucos comentários, as porcentagens mudam muito com um comentário a mais ou a menos | Mostra o painel. Não há aviso de amostra pequena, só o selo "Poucos comentários" nos vídeos com menos de 5 |
| **Marca em vez de produto** (real): Samsung, apelido `sg` | 5 vídeos, 148 comentários, só 17 (11%) citam a marca | O público fala do modelo (A57, S25), não da marca. Apelidos curtos como `sg` podem casar com outras palavras. O NER listou um nome próprio entre as marcas | Mostra o painel, mas a maioria dos comentários fica em "Demais comentários" |
| **Cota esgotada no meio da coleta** (simulada) | O app mantém o que já tinha coletado (5 vídeos, 60 comentários no teste) | A API responde `quotaExceeded` | Aviso "A cota diária da API acabou no meio da coleta: o resultado está incompleto" |
| **Cota já esgotada** (simulada) | 0 vídeos | Nenhuma chamada passa | O aviso de cota e a mensagem de nenhum comentário. As análises salvas continuam abrindo |
| **Vídeo com comentários desativados** (simulada) | O vídeo é ignorado sem quebrar a coleta | A API responde `commentsDisabled` | Nenhuma mensagem; o vídeo só não aparece |
| **Sem chave no `.env`** (código) | A análise não começa | `YOUTUBE_API_KEY` ausente | Erro explicando como criar o `.env` e sugerindo abrir uma análise salva |
| **Classificador não treinado** (código) | A análise não começa | Falta `bertimbau_congelado_rl.joblib` | Erro com o comando `python src/treinar_classificador.py` |
| **Campos incompletos** (código) | — | Produto vazio ou período com uma data só | Aviso pedindo o produto ou a data final |

**O que aprendemos com as falhas**
- O app funciona bem para produtos com muitos vídeos e comentários em português.
- Para marcas pequenas, infantis ou com nome em inglês, o YouTube devolve pouco material. Os filtros de vídeo infantil e de idioma descartam o resto.
- Quando há pouco material, a resposta certa é avisar, e não inventar um resultado.
- Ficaram como melhorias:
  - avisar quando a amostra for pequena;
  - sugerir ampliar o período automaticamente;
  - tirar termos genéricos ("celular", números soltos) da lista de marcas.

## Dados, licenças e ética

- **RePro:** 10 mil avaliações da Americanas, extraídas da B2W-Reviews01 e rotuladas por humanos. A licença é CC BY-NC-SA 4.0, ou seja, só para uso não comercial.
- **Comentários do YouTube:**
  - Coletados pela API oficial.
  - O cliente descarta nome, foto e canal do autor (`_anonimizar` em `youtube.py`).
  - Os textos ficam só na máquina local, fora do git. No repositório vão apenas IDs e rótulos.
- **Vieses:**
  - O classificador foi treinado em avaliações de loja e erra mais no YouTube, principalmente com ironia e com o neutro.
  - O resultado do app é um indicador, não uma medida exata da opinião do público.
- **Classificador do app:** foi treinado no RePro, então segue a mesma licença, CC BY-NC-SA 4.0.
- **Fora do git:**
  - `.env`, `.venv`, `modelos_baixados/` e `docs/raw/`;
  - `src-pln/models/`, exceto o classificador do app;
  - `src-pln/data/embeddings/`, `src-pln/coleta_youtube/dados/` e `src/analises/`.
