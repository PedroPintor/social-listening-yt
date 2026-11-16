> **Nota (16/11/2026).** Esta é a proposta entregue na etapa 1, em 22/10/2026, sem alterações no texto. Ao longo do projeto, algumas escolhas mudaram:
>
> | Na proposta | O que fizemos | Por quê / onde está |
> |---|---|---|
> | NER com spaCy (`pt_core_news_lg`) + EntityRuler com marcas da B2W | GLiNER2 PT-BR (`ottema/gliner2-ptbr`), com categorias livres | O spaCy em português não tem a classe produto. Ver a etapa 3 do [`src-pln/README.md`](../src-pln/README.md) e o notebook 03 |
> | B2W-Reviews01 como catálogo de marcas e dados extras | Não usada | O RePro já é a parte da B2W rotulada por humanos (sobreposição); a nota em estrelas não é sentimento; o GLiNER2 dispensou o catálogo |
> | Pseudo-rotulagem da B2W na rede neural | Não feita | Saiu junto com a B2W. A rede neural compara BiLSTM, BiGRU e CNN, com e sem embeddings pré-treinados (notebook 04) |
> | NER para confirmar quais comentários citam o produto | Busca pelo nome do produto e apelidos (Precision 0,91); o NER só lista as marcas citadas | O NER como filtro ficou com Precision ~0,50 em qualquer limiar (notebook 05c) |
> | Ajuste fino do BERTimbau | Ajuste fino e também o BERTimbau congelado + Regressão Logística, que foi o melhor no YouTube (F1 0,807) | Notebooks 05 e 05b |
> | Amostra de 200 a 300 comentários "só para teste" | 350 textos (300 comentários e 50 títulos), usados como validação, kappa 0,48 | [README da coleta](../src-pln/coleta_youtube/README.md) e notebook 05b |
> | Chatbot | Painel em Streamlit, sem chatbot | [`docs/notas/decisoes_de_escopo.md`](notas/decisoes_de_escopo.md) |
> | P/R/F1 por entidade nos comentários reais | Feito só em 24 frases sintéticas | A rotulagem manual não anotou entidades; o NER real foi avaliado como filtro (05c) |

---

# Monitoramento de Marca no YouTube: Análise de Sentimentos e Reconhecimento de Entidades em Comentários em Português

Oct 26, 2026 · @Pedro Pintor

## Identificação

**Integrantes:** Pedro Garcia Pintor e Maria Alice.

**Tema:** monitoramento de marca em comentários do YouTube em português, combinando análise de sentimentos e reconhecimento de entidades nomeadas (NER).

## Problema

Empresas não conseguem acompanhar manualmente o que o público diz sobre seus produtos no YouTube. Um único produto, como um brinquedo, pode aparecer em dezenas de vídeos de criadores independentes, com milhares de comentários espalhados.

O projeto responde à pergunta: **"Como a empresa X pode saber se, no período Z, o produto Y foi mencionado no YouTube e se as menções foram positivas, negativas ou neutras?"**

Há também um desafio técnico. Não existe uma base pública de comentários do YouTube em português rotulados por sentimento. Verificamos uma base do Kaggle com cerca de 1 milhão de comentários rotulados, e menos de 0,3% deles estavam em português. Por isso, os modelos serão treinados com avaliações de produtos de e-commerce e depois aplicados ao YouTube, um problema conhecido como **mudança de domínio**.

## Justificativa

O YouTube é uma das principais fontes de opinião sobre produtos, e a percepção expressa nos comentários influencia a decisão de compra de outros consumidores. Ferramentas comerciais de monitoramento de marca (*social listening*) existem, mas são pagas e focadas em inglês.

O projeto também tem interesse acadêmico: medir quanto um modelo treinado em avaliações de loja perde ao ser aplicado em comentários informais do YouTube, que têm gírias, ironia, emojis e textos curtos.

## Público-alvo e objetivos

**Público-alvo:** equipes de marketing, atendimento e gestão de produto de empresas que querem acompanhar a percepção de suas marcas e produtos no YouTube.

**Objetivo geral:** desenvolver uma aplicação em que o usuário informa uma marca ou produto e um período, e recebe um resumo das menções no YouTube e do sentimento associado a elas.

**Objetivos específicos:**

1. Classificar o sentimento de comentários em português em positivo, negativo e neutro.
2. Reconhecer marcas, produtos, organizações e pessoas mencionadas nos comentários (NER).
3. Comparar abordagens clássicas, redes neurais e Transformers pré-treinados.
4. Medir a perda de desempenho na troca de domínio (e-commerce → YouTube), usando uma amostra de comentários rotulada manualmente pelo grupo.
5. Disponibilizar um chatbot que responda perguntas sobre os resultados com base apenas nos dados processados.

## Dados

O projeto usa duas bases públicas em português para treino e avaliação, e dados do YouTube para a aplicação.

| Fonte | Conteúdo | Uso no projeto | Onde baixar | Licença |
| --- | --- | --- | --- | --- |
| RePro (PROPOR 2024) | 10.000 avaliações da Americanas.com (cerca de 2.000 por nota de estrela), extraídas da B2W-Reviews01 e rotuladas por anotadores humanos: sentimento (positivo, negativo, neutro, misto) e tópico (produto, entrega etc.); concordância kappa de 0,71 | Treino e avaliação principal do classificador de sentimentos | [GitHub](https://github.com/lucasnil/repro) · [Hugging Face](https://huggingface.co/datasets/lucasnil/repro) | CC BY-NC-SA 4.0 |
| B2W-Reviews01 (STIL 2019) | \~132 mil avaliações da Americanas.com (2018) com nome do produto, marca, categoria, texto e nota de 1 a 5, sem rótulo manual de sentimento | Nomes de produtos e marcas para o NER; dados extras para a rede neural | [GitHub](https://github.com/americanas-tech/b2w-reviews01) | CC BY 4.0 |
| YouTube Data API v3 | Títulos de vídeos e comentários, filtrados por termo de busca e período | Aplicação final e amostra de teste fora do domínio | Coleta pela API | Termos de uso do YouTube |

**Amostra de avaliação do YouTube:** o grupo vai coletar e rotular manualmente cerca de 200 a 300 comentários sobre produtos. Os dois integrantes rotulam a mesma amostra, e a concordância entre eles será medida pelo kappa de Cohen. Essa amostra serve só para teste, nunca para treino.

**Classes de sentimento:** positivo, negativo e neutro. Os casos "misto" do RePro serão analisados à parte.

## Técnicas de PLN e tecnologias previstas

| Etapa | Técnica prevista | Ferramentas |
| --- | --- | --- |
| Análise de sentimentos | Pré-processamento + TF-IDF com Regressão Logística como linha de base | scikit-learn, NLTK |
| NER | Modelo pré-treinado em português (pessoas, locais, organizações) adaptado com uma lista de marcas e produtos extraída da B2W | spaCy (pt\_core\_news\_lg), EntityRuler |
| Rede neural | LSTM bidirecional com embeddings de palavras; teste de pseudo-rotulagem, usando o Transformer para rotular dados extras da B2W | PyTorch ou Keras |
| Transformers | Ajuste fino (*fine-tuning*) do BERTimbau e comparação com um modelo multilíngue de sentimento já pronto | Hugging Face Transformers (neuralmind/bert-base-portuguese-cased, cardiffnlp/twitter-xlm-roberta-base-sentiment) |
| Aplicação / chatbot | Busca na API, filtragem das menções por NER, classificação e respostas baseadas nos resultados | YouTube Data API v3, Streamlit ou Gradio |

**Métricas:** acurácia, precisão, revocação e F1 (macro) para sentimento; precisão, revocação e F1 por entidade para NER. Todas serão acompanhadas de análise qualitativa dos erros.

**Linguagem e ambiente:** Python, Jupyter/Google Colab e GitHub para versionamento.

## Integração e aspectos éticos

**Como os componentes se conectam:** o usuário informa a marca ou o produto e o período → a aplicação busca vídeos e comentários na API do YouTube → o NER confirma quais comentários mencionam o alvo → o melhor classificador (escolhido nas comparações entre abordagens) atribui o sentimento → o chatbot responde perguntas como "Quais foram as principais reclamações sobre o produto Y na última semana?".

**Aspectos éticos:** nomes e IDs dos autores dos comentários serão removidos ou anonimizados. As licenças das bases serão respeitadas (o RePro permite apenas uso não comercial, o que é compatível com um trabalho acadêmico). O relatório vai discutir vieses dos modelos, como dificuldade com ironia e gírias, e o risco de decisões empresariais baseadas em classificações automáticas.

## Referências

- SILVA, L. N. S. *et al.* [RePro: A Benchmark Dataset for Opinion Mining in Brazilian Portuguese](https://aclanthology.org/2024.propor-1.44/). In: *Proceedings of the 16th International Conference on Computational Processing of Portuguese (PROPOR 2024)*, v. 1, p. 432–440, 2024.
- REAL, L.; OSHIRO, M.; MAFRA, A. [B2W-Reviews01: an open product reviews corpus](https://github.com/americanas-tech/b2w-reviews01). In: *Symposium in Information and Human Language Technology (STIL)*, 2019.
