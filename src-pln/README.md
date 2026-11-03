
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
- `src-pln/notebooks/etapa_analise_sentimentos.ipynb` — notebook da análise de sentimentos.
- `src-pln/requirements.txt` — dependências do projeto.

### Resultado da etapa

Foi implementado um classificador inicial de sentimentos baseado em TF-IDF e Regressão Logística, que servirá como referência para os experimentos posteriores.

## Etapa 3 — Reconhecimento de Entidades Nomeadas (NER)

**Status:** próxima etapa de desenvolvimento.

O objetivo será identificar automaticamente marcas, produtos, organizações e pessoas mencionadas nos textos.

### Abordagem prevista

- Utilizar o spaCy com um modelo pré-treinado em português.
- Complementar o reconhecimento com o `EntityRuler`.
- Utilizar nomes de marcas e produtos da base B2W-Reviews01.
- Testar a extração de entidades em exemplos reais.
- Analisar entidades não reconhecidas e classificações incorretas.

## Próximas etapas

- **Etapa 4:** Redes neurais para PLN.
- **Etapa 5:** Transformers e modelos pré-treinados.
- **Etapa 6:** Chatbot ou aplicação multimodal.
- **Etapa 7:** Integração, documentação e apresentação final.
