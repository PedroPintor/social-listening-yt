
# Social Listening — Monitoramento de Marca no YouTube

## Sobre o projeto

Este projeto tem como objetivo desenvolver uma solução de monitoramento de marcas e produtos em comentários do YouTube, utilizando técnicas de Processamento de Linguagem Natural (PLN).

A aplicação proposta permitirá identificar menções a marcas e produtos, classificar sentimentos e analisar a percepção dos consumidores.

**Integrantes:** Pedro Garcia Pintor e Maria Alice.

## Etapa 1 — Definição do problema

Foram definidos o problema, o público-alvo, os objetivos, as fontes de dados e as tecnologias previstas para o desenvolvimento da solução.

**Objetivo:** permitir que uma empresa identifique se determinado produto ou marca foi mencionado no YouTube durante um período e se essas menções foram positivas, negativas ou neutras.

## Etapa 2 — Análise de sentimentos

Foi desenvolvido um experimento de classificação de sentimentos utilizando a base **RePro**, composta por 10.000 avaliações de produtos da Americanas.com, rotuladas por anotadores humanos.

### O que o código faz?

1. **Carregamento dos dados:** importa as avaliações da base RePro.
2. **Preparação dos textos:** organiza os comentários e seus rótulos, separando os casos de sentimentos mistos.
3. **Vetorização:** transforma os textos em representações numéricas utilizando TF-IDF.
4. **Treinamento:** utiliza Regressão Logística para classificar sentimentos positivos, negativos e neutros.
5. **Avaliação:** calcula métricas de classificação, como acurácia e F1-score, e apresenta uma matriz de confusão.
6. **Testes:** realiza previsões em novos comentários, incluindo exemplos no estilo YouTube.
7. **Análise de erros:** identifica classificações incorretas e discute as limitações do modelo.

### Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Jupyter Notebook

### Arquivos principais

- `src-pln/data/RePro.csv` — base de avaliações de produtos.
- `src-pln/notebooks/02_etapa_analise_sentimentos.ipynb` — notebook da análise de sentimentos.
- `src-pln/requirements.txt` — dependências do projeto.

### Resultado da etapa

Foi implementado um classificador inicial de sentimentos baseado em TF-IDF e Regressão Logística, que servirá como referência para os experimentos posteriores.

## Etapa 3 — Reconhecimento de Entidades Nomeadas (NER)

**Status:** concluída.

O objetivo é identificar automaticamente marcas, produtos, empresas e pessoas mencionadas nos textos.

### Modelo: GLiNER2 PT-BR (e não spaCy)

Foi utilizado o [GLiNER2 PT-BR](https://huggingface.co/ottema/gliner2-ptbr) (`ottema/gliner2-ptbr`, pacote `gliner2`). A ideia inicial era usar o spaCy, mas o modelo em português do spaCy só reconhece `PER`, `ORG`, `LOC` e `MISC` e **não tem a classe produto**, que é central para o monitoramento de marca. O GLiNER2 aceita **categorias livres** passadas como texto (por exemplo, `"marca"`, `"produto"`), sem precisar treinar o modelo de novo.

### O que o código faz?

1. **Carregamento do modelo** pré-treinado do Hugging Face.
2. **Teste exploratório** em uma amostra aleatória de 10 avaliações da RePro, comparando com a marca e o produto cadastrados.
3. **Comparação de categorias:** simples (`marca`, `produto`, `empresa`, `pessoa`) × específicas (`marca comercial`, `nome de produto`...).
4. **Comentários no estilo YouTube** e teste do parâmetro `threshold` (limiar de confiança).
5. **Mini-avaliação quantitativa:** 24 comentários sintéticos com gabarito anotado à mão (42 entidades), com precisão, revocação e F1 por categoria.
6. **Análise de erros** e das limitações do modelo.

### Resultado da etapa

Na mini-avaliação sintética, o GLiNER2 teve **F1 de 0,597** com correspondência exata de texto e categoria (precisão 0,451, revocação 0,881) e **F1 de 0,857** quando a categoria é ignorada. O modelo encontra quase todas as menções, mas repete o mesmo trecho em várias categorias (marca e empresa, principalmente) e confunde gírias como "mano" com pessoas. Aumentar o `threshold` quase não ajuda, porque as confianças ficam perto de 1,0. A avaliação é pequena e sintética; a avaliação definitiva será feita com comentários reais do YouTube rotulados manualmente.

### Arquivos principais

- `src-pln/notebooks/03_reconhecimento_entidades.ipynb` — notebook do reconhecimento de entidades.

## Etapa 4 — Rede neural para análise de sentimentos

**Status:** concluída.

A rede neural é o degrau do meio entre o TF-IDF (etapa 2) e o BERTimbau (etapa 5). A tarefa, os textos e o conjunto de teste são os mesmos da etapa 2. A preparação dos dados foi movida para `src/dados.py`, usado pelos dois notebooks. A pergunta é quanto valem **(a)** ler a ordem das palavras e **(b)** o conhecimento prévio dos embeddings pré-treinados.

### O que o código faz?

1. **Mesmos dados e mesmo teste da etapa 2**, com validação estratificada de 15% do treino para o *early stopping*.
2. **Tokenização simples** (minúsculas, mantendo acentos, pontuação e emojis), tamanho máximo pelo percentil 95 (80 tokens) e vocabulário com `min_freq=2`.
3. **Fase 1:** BiLSTM × BiGRU × CNN 1D (Kim, 2014), com embeddings aprendidos do zero, pesos por classe, dropout e *early stopping* pelo F1 macro, com 3 sementes cada.
4. **Fase 2:** a vencedora (BiGRU) com embeddings **NILC GloVe 300d** e **FastText cc.pt.300**, congelados e ajustados, com cobertura do vocabulário de cada um.
5. **Avaliação:** tabela comparativa com o TF-IDF (métricas, parâmetros, tempos), curvas, matriz de confusão, frases no estilo YouTube, mistos e análise de erros.

### Resultado da etapa (teste, média de 3 sementes)

| Modelo | Acurácia | F1 macro | F1 neutro | Inferência |
|---|---|---|---|---|
| TF-IDF + Regressão Logística (etapa 2) | 0,951 | 0,883 | 0,724 | 0,4 ms |
| BiGRU, embeddings do zero | 0,942 | 0,868 | 0,696 | ~3 ms |
| **BiGRU + NILC GloVe (ajustado)** | **0,957** | **0,892** | **0,739** | ~3,5 ms |

As redes **do zero perderam** para o TF-IDF (as três arquiteturas empataram entre si). Com **embeddings pré-treinados do NILC ajustados**, a rede passou o baseline por pouco, mas de forma consistente nas 3 sementes. Ela corrigiu negações ("não tem igual entre os intermediários") e gírias ("top demais"), mas não a ironia. O ganho veio do **conhecimento prévio, e não da arquitetura**, o que reforça o uso do BERTimbau na etapa 5. Congelar os embeddings e usar o FastText do Facebook deram resultados piores.

### Arquivos principais

- `src-pln/notebooks/04_rede_neural.ipynb` — notebook experimental.
- `src-pln/src/dados.py` — preparação do RePro (igual à da etapa 2) e tokenização; `src-pln/src/redes.py` — vocabulário, arquiteturas e carregamento do modelo (`ClassificadorSentimento`).
- `src-pln/utils/baixar_embeddings.py` — baixa e filtra os embeddings para `data/embeddings/` (fora do git): `python utils/baixar_embeddings.py`.
- `src-pln/reports/04_resultados_rede_neural.csv` e `04_execucoes_por_semente.csv` — resultados.
- `src-pln/models/sentimento_rede_neural.pt` — melhor modelo (63 MB, fora do git; é recriado ao executar o notebook).

## Próximas etapas

- **Etapa 5:** Transformers e modelos pré-treinados.
- **Etapa 6:** Chatbot ou aplicação multimodal.
- **Etapa 7:** Integração, documentação e apresentação final.
