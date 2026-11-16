# Radar de Marca: monitoramento de marca em comentários do YouTube com PLN

**Relatório técnico, Processamento de Linguagem Natural, novembro de 2026**
**Integrantes:** Pedro Garcia Pintor e Maria Alice
**Repositório:** <https://github.com/PedroPintor/social-listening-yt>

---

## 1. Introdução

**Problema.** Empresas não conseguem acompanhar à mão o que o público diz de seus produtos no YouTube. Um mesmo produto aparece em dezenas de vídeos de criadores independentes, cada um com centenas de comentários. As ferramentas comerciais de *social listening* existem, mas são pagas e pensadas para o inglês.

**Pergunta do projeto.** *Como a empresa X pode saber se, no período Z, o produto Y foi mencionado no YouTube e se as menções foram positivas, negativas ou neutras?*

**Contexto e justificativa.** Não existe uma base pública de comentários do YouTube em português rotulados por sentimento. Numa base do Kaggle com cerca de 1 milhão de comentários rotulados, menos de 0,3% estavam em português. Por isso, os modelos foram treinados em avaliações de e-commerce e aplicados ao YouTube. Esse problema se chama **troca de domínio**: os textos de treino e os textos de uso vêm de lugares diferentes. Medir quanto se perde nessa troca é o interesse acadêmico do trabalho. O interesse prático é entregar uma ferramenta que funcione.

**Objetivos.**
1. Classificar o sentimento de comentários em português (positivo, neutro, negativo).
2. Reconhecer marcas e produtos citados (NER).
3. Comparar abordagens clássicas, redes neurais e Transformers.
4. Medir a perda na troca de domínio com uma amostra do YouTube rotulada pelo grupo.
5. Integrar tudo numa aplicação: o **Radar de Marca**.

A proposta original e o que mudou depois estão em [`docs/proposta.md`](../../docs/proposta.md).

## 2. Fundamentação teórica

Os conceitos desta seção seguem o material da disciplina. As citações indicam os dados e os modelos pré-treinados que usamos.

**TF-IDF + Regressão Logística.** O texto vira um vetor esparso. Cada posição é uma palavra ou par de palavras, com peso maior para termos frequentes no texto e raros no corpus. A Regressão Logística aprende um peso por termo. O modelo é rápido e interpretável, mas trata o texto como um "saco de palavras", sem ordem e sem contexto.

**Redes recorrentes (BiGRU).** A GRU lê o texto palavra por palavra e mantém um estado oculto. *Portões* aprendidos decidem o que guardar e o que esquecer. A versão bidirecional lê nos dois sentidos, então cada palavra "vê" o que vem antes e depois. Comparamos também a BiLSTM e uma CNN 1D. As palavras entram como *embeddings*, vetores densos que podem ser aprendidos do zero ou vir pré-treinados em grandes corpora. Usamos o GloVe do NILC (Hartmann et al., 2017) e o FastText do Facebook (Grave et al., 2018). Esses vetores são **estáticos**: "bateria" tem o mesmo vetor em qualquer frase.

**Transformers.** O Transformer substitui a recorrência pela **autoatenção**: cada palavra calcula quanto deve olhar para cada uma das outras, e o vetor dela passa a depender da frase inteira. Por isso os vetores são **contextuais**. O BERT é um codificador Transformer pré-treinado em texto cru, prevendo palavras mascaradas. O **BERTimbau** (Souza et al., 2020) é o BERT treinado no brWaC, um corpus da web brasileira. A versão base tem 12 camadas e ~110 milhões de parâmetros. Há dois modos de uso:
- **congelado:** os pesos não mudam; a média dos vetores da última camada representa o texto, e um classificador simples é treinado em cima;
- **ajustado (*fine-tuning*):** todos os pesos são atualizados na tarefa.

O **XLM-R de tweets** da Cardiff NLP (Barbieri et al., 2022) é um XLM-RoBERTa já ajustado para sentimento em tweets de 8 línguas, incluindo o português. Ele foi usado sem nenhum treino nosso (*zero-shot*).

**NER com GLiNER2.** O reconhecimento de entidades nomeadas marca trechos do texto com uma categoria. Modelos clássicos, como o spaCy em português, têm um conjunto fixo de categorias (`PER`, `ORG`, `LOC`, `MISC`) e não têm *produto*. O GLiNER2 recebe as categorias **como texto** no momento do uso ("marca", "produto"). O modelo compara o trecho com a descrição da categoria, sem precisar de retreino. Usamos o `ottema/gliner2-ptbr`, ajustado para o português.

**Avaliação.**
- **F1 macro:** média do F1 de cada classe; dá o mesmo peso ao neutro, que é raro, e às classes grandes.
- **Kappa de Cohen:** concordância entre dois anotadores descontando a concordância por acaso: κ = (pₒ − pₑ)/(1 − pₑ). Interpretamos pela escala de Landis e Koch (1977).
- **Bootstrap:** reamostra o conjunto de teste milhares de vezes para obter um intervalo de confiança da diferença de F1.
- **Teste de McNemar:** olha só os itens em que um modelo acerta e o outro erra, e testa se essa divisão poderia ser acaso.

## 3. Dados e metodologia

### 3.1 Fontes

| Fonte | Conteúdo | Uso |
|---|---|---|
| **RePro** (Silva et al., 2024) | 10.000 avaliações da Americanas.com, extraídas da B2W-Reviews01 e rotuladas por anotadores humanos (kappa 0,71). Licença CC BY-NC-SA 4.0 | Treino e teste de todos os classificadores |
| **Amostra do YouTube** (nossa) | 350 textos reais (300 comentários e 50 títulos) de 6 produtos, rotulados pelos dois integrantes | **Validação** fora do domínio; nunca usada para treinar |
| **Frases sintéticas** (nossas) | 24 comentários no estilo YouTube com 42 entidades anotadas | Avaliação do NER |

A **B2W-Reviews01** (132 mil avaliações com nota em estrelas) estava prevista na proposta como catálogo de marcas para o NER e como dado extra para a rede neural, por pseudo-rotulagem. Ela foi descartada por três motivos:
- o RePro já é a parte da B2W rotulada por humanos, o que traria sobreposição;
- a nota em estrelas não é sentimento;
- o GLiNER2 dispensou o catálogo de marcas.

### 3.2 Preparação do RePro

Cada avaliação tem uma lista de polaridades. As que têm **um** sentimento formam o conjunto de 3 classes:

| Classe | Avaliações |
|---|---|
| Positivo | 4.127 |
| Negativo | 3.449 |
| Neutro | 409 (5%) |
| **Total** | **7.985** |

As 2.018 avaliações **mistas** ficaram fora do treino e foram analisadas à parte.

O texto é o título mais a avaliação. Para os modelos clássicos e para a rede neural, ele passa a minúsculas, sem links e sem números. Os Transformers recebem o texto original.

A divisão é 80/20, estratificada e com semente 42: **6.388 avaliações para treino e 1.597 para teste**, iguais em todas as etapas. Os hiperparâmetros foram escolhidos numa validação retirada do treino: 15% para a rede neural e 10% para os Transformers. Os modelos com variação aleatória foram treinados com 3 sementes.

### 3.3 Amostra do YouTube e rotulagem

**Coleta.** Em 09/11/2026, pela YouTube Data API v3, coletamos vídeos publicados entre 02 e 08/11 de seis produtos: iPhone 17, PlayStation 5, BYD Dolphin Mini, iFood, Havaianas e O Boticário. O resultado foram 101 vídeos. Desses vídeos, sorteamos de forma estratificada 300 comentários e 50 títulos.

**Rotulagem.**
- Os dois integrantes rotularam a amostra **de forma independente**, com uma ferramenta própria (`rotulador.html`).
- Havia cinco rótulos: positivo, neutro, negativo, misto e **não é sobre o produto**.
- O quinto rótulo surgiu na prática: no YouTube, muitos comentários falam do vídeo, do youtuber ou de outra coisa.

**Gabarito.** Só entraram os itens em que os dois concordaram. Não forçamos consenso nos itens em que discordamos, para não inventar uma "verdade" onde nem humanos concordam. O gabarito resultante tem:
- **86 comentários** com sentimento: a base das métricas de sentimento;
- **88 comentários** "não é sobre o produto": a base da análise de relevância;
- **40 títulos** com sentimento.

### 3.4 Desenho experimental

1. **Etapas 2 a 5:** todos os modelos são treinados no treino do RePro e comparados no mesmo teste.
2. **Validação no YouTube (`05b`):** os mesmos modelos, sem nenhum ajuste, são aplicados aos 86 comentários e aos 40 títulos. A queda de F1 mede a troca de domínio.
3. **Fluxo do app (`05c`):** testamos como separar os comentários sobre o produto (NER × busca pelo nome) e a significância estatística das diferenças entre os modelos.

## 4. Implementação

O **Radar de Marca** é um app em Streamlit (`src/app.py`). O fluxo fica em `src/pipeline.py`:

```
produto + apelidos + período
   → 1. Coleta (YouTube Data API: busca de vídeos, detalhes, comentários; cache em disco)
   → 2. Separação: o comentário cita o nome do produto ou um apelido?
   → 3. Sentimento: BERTimbau congelado + Regressão Logística (positivo / neutro / negativo)
   → 4. NER: GLiNER2 extrai marcas, produtos e empresas citadas
   → Painel: métricas, barras de sentimento por grupo, cards dos vídeos, lista de comentários, marcas citadas
```

**1. Coleta** (`src-pln/coleta_youtube/youtube.py`).
- Faz a busca por região BR e idioma pt.
- Descarta vídeos infantis (`madeForKids`), vídeos sem comentários e vídeos com áudio declarado em outra língua.
- Remove nome, foto e canal do autor antes de salvar qualquer coisa.
- Guarda cada resposta em cache e conta as unidades gastas da cota diária (10.000). Uma análise gasta cerca de 100 a 120 unidades.
- Quando a cota acaba, mantém o que já coletou e avisa.

**2. Separação.** É uma busca, sem acento, do nome e dos apelidos que o usuário digita (aceita plural simples). **Nenhum comentário é descartado.** O painel mostra o sentimento de dois grupos: "citam o produto" e "demais comentários". A escolha está justificada na seção 5.4.

**3. Sentimento** (`src/sentimento.py`).
- Cada comentário vira a média dos vetores da última camada do BERTimbau (768 dimensões, até 128 *tokens*).
- Uma Regressão Logística (C = 0,1, classes balanceadas) dá a classe.
- Só a Regressão Logística foi treinada por nós: são 2.307 parâmetros, salvos num arquivo de 10 KB que vai no repositório.

**4. NER.** O GLiNER2 extrai entidades das categorias "marca", "produto" e "empresa", com limiar 0,5. O painel conta em quantos comentários cada entidade aparece e destaca as que não são o produto pesquisado. Assim a empresa descobre **com quem é comparada**.

**Painel.**
- Cada vídeo recebe um selo: positivo ou negativo quando um lado supera o outro em 10 pontos percentuais, e "dividido" nos demais casos.
- As análises são salvas em JSON e podem ser reabertas sem gastar cota.
- O BERTimbau e o GLiNER2 são baixados do Hugging Face na primeira execução, para a pasta `modelos_baixados/`.
- As instruções de execução e o registro de falhas estão no [README](../../README.md).

**Por que um painel e não um chatbot.** O passo a passo sugere um chatbot. Optamos por um painel, com o aceite do professor, por três motivos:
- **Perguntas fechadas:** "quanto é negativo?", "quais vídeos puxam a conversa?" e "com quem me comparam?" o painel já responde com números exatos, numa tela só.
- **Perguntas abertas:** "por que este vídeo ficou negativo?", "quais as principais reclamações?" e "o que o youtuber disse do produto?" exigiriam um LLM com RAG (geração aumentada por recuperação) sobre os comentários **e** sobre a transcrição do vídeo. A API não fornece essa transcrição para vídeos de terceiros, e essa seria uma segunda frente de trabalho, com custo e avaliação próprios.
- **Etapa 6:** o painel cumpre o que ela pede: uma interface que integra os modelos e apresenta resultados compreensíveis.

## 5. Experimentos e avaliação

### 5.1 Concordância humana: quão difícil é a tarefa

Antes de medir os modelos, medimos a própria tarefa.

**Kappa geral (5 classes): κ = 0,48, concordância moderada.** Os dois concordaram em 216 dos 350 itens (pₒ = 0,617). A concordância esperada por acaso era pₑ = 0,26, então κ = (0,617 − 0,26)/(1 − 0,26) = 0,48.

| Recorte | Kappa | Interpretação (Landis e Koch) |
|---|---|---|
| Todos os itens, 5 classes | 0,48 | Moderada |
| Só a polaridade (positivo, neutro, negativo), nos itens que os dois consideraram sobre o produto | 0,63 | Substancial |
| Títulos | 0,36 | Razoável |
| Anotadores do RePro, na loja | 0,71 | Substancial |

O que esses números mostram:
- **A maior divergência estava em decidir *se* o comentário é sobre o produto.** Pedro marcou "não é sobre o produto" em 64 itens aos quais Maria Alice deu um sentimento.
- **Nos títulos, faltava um ponto de vista.** 92% (Pedro) e 74% (Maria Alice) foram rotulados como neutros. Um título como "iPhone 17 vs S25: qual comprar?" só tem sentimento do ponto de vista de alguém.
- **O YouTube é mais difícil do que a loja até para humanos:** 0,48 contra 0,71.

Por isso, nenhum modelo deve ser avaliado esperando acerto perfeito.

### 5.2 NER

Avaliação nas 24 frases sintéticas (`03`), com correspondência exata do trecho:

| Categoria | Precision | Recall | F1 |
|---|---|---|---|
| marca | 0,62 | 1,00 | 0,76 |
| produto | 0,42 | 0,67 | 0,51 |
| empresa | 0,28 | 1,00 | 0,44 |
| pessoa | 0,57 | 1,00 | 0,73 |
| **Total, trecho e categoria** | 0,45 | 0,88 | **0,60** |
| **Total, ignorando a categoria** | 0,80 | 0,93 | **0,86** |

**O que o GLiNER2 faz bem.** Ele **acha** quase todas as menções (Recall 0,88), inclusive apelidos como "Play 5".

**O que ele faz mal.**
- **Classifica mal:** o mesmo trecho sai como marca *e* como empresa, e a categoria *produto* é a pior.
- **Gírias viram pessoas:** palavras como "mano" são marcadas como pessoa.
- **O limiar quase não ajuda:** as confianças ficam perto de 1.

Nos comentários reais, vimos os mesmos tipos de erro:
- "Brasil", em "Tropa do Brasil nunca ganhei um like", saiu como marca;
- "17 Lilac", em "Eu assistindo no meu 17 Lilac ❤", não foi reconhecido como produto.

**Limitação desta avaliação.** A rotulagem do YouTube não anotou entidades, então não há P/R/F1 por entidade em dados reais. Nos dados reais, o NER foi avaliado como **filtro de relevância** (5.4).

### 5.3 Sentimento: RePro e YouTube

Cada modelo testa uma hipótese:

| Modelo | Hipótese | Ponto forte | Ponto fraco |
|---|---|---|---|
| TF-IDF + RL | Palavras soltas bastam? | Rápido, interpretável | Sem ordem nem contexto |
| BiGRU + GloVe | Ler a sequência ajuda? | Capta ordem e negação local | Vetores estáticos; precisa de muitos dados |
| XLM-R de tweets | Um modelo de rede social se adapta melhor ao YouTube? | Já conhece texto informal | Nunca viu o nosso domínio; grande e lento |
| BERTimbau congelado + RL | O contexto do português ajuda, sem especializar na loja? | Vetores contextuais; só 2.307 parâmetros treinados | Não se adapta à tarefa |
| BERTimbau ajustado | Ajustar à tarefa ajuda? | O melhor na loja | Pode especializar no estilo da loja |

**Rede neural (`04`, teste do RePro, média de 3 sementes).**
- As três arquiteturas treinadas do zero empataram entre si e perderam para o TF-IDF: BiLSTM 0,870, BiGRU 0,868, CNN 0,866, contra 0,883.
- Com o GloVe do NILC ajustado, a BiGRU chegou a 0,892, acima do TF-IDF nas 3 sementes.
- O FastText e os embeddings congelados foram piores.
- Conclusão: o ganho veio do **conhecimento prévio** dos embeddings, não da arquitetura.

**Transformers (`05`).** O ajuste fino usou taxa de aprendizado 2e-5, lote de 32, 3 épocas, aquecimento de 10% e *weight decay* de 0,01, com 3 sementes, no Google Colab com GPU.

**Resultados (F1 macro):**

| Modelo | RePro (1.597) | YouTube (86) | Queda | F1 neutro YouTube | Erro na proporção* | CPU, 300 comentários |
|---|---|---|---|---|---|---|
| TF-IDF + RL | 0,883 | 0,744 | −0,14 | 0,667 | 16% | 0,1 s |
| BiGRU + GloVe** | 0,887 | 0,739 | −0,15 | 0,595 | 11% | ~3 s |
| XLM-R tweets | 0,805 | 0,774 | **−0,03** | 0,667 | **6%** | ~60 s |
| **BERTimbau congelado + RL** | 0,886 | **0,807** | −0,08 | **0,727** | 8% | ~50 s |
| BERTimbau ajustado | **0,921** | 0,788 ± 0,031 | −0,13 | 0,671 | 11% | ~50 s |

\* É a maior diferença entre a porcentagem prevista e a real de positivo, neutro ou negativo. Mede o erro do **resumo** que o app mostra, e não o de cada comentário. Por exemplo, o TF-IDF diria 53% de negativos quando o real é 37%.
\*\* Modelo salvo (semente 42). A média das 3 sementes no RePro é 0,892.

**Leitura dos resultados.**
- **A ordem muda de um domínio para o outro.** Na loja, o ajustado vence. No YouTube, o congelado vence e o ajustado cai mais (−0,13): ajustar **especializou** o modelo no estilo das avaliações de loja.
- **O XLM-R confirma a hipótese do domínio.** É o pior na loja e o que menos cai no YouTube (−0,03). Por isso tem o menor erro na proporção.
- **Títulos.** Nos 40 títulos, todos os modelos ficam entre 0,17 e 0,49 de F1. Como os títulos são quase todos neutros, o app não atribui sentimento a eles.

**Exemplos reais** (gabarito → previsão):

| Comentário (produto) | Gabarito | BERTimbau cong. | TF-IDF | O que mostra |
|---|---|---|---|---|
| "Estou com um 17e há 2 meses e posso afirmar que é um excelente aparelho. […] Nunca uso a ultrawide" (iPhone) | positivo | ✅ positivo | ❌ negativo | Negação: "nunca" puxa o TF-IDF para negativo |
| "Lá eles não tem chinelo?" (Havaianas) | neutro | ✅ neutro | ❌ negativo | Pergunta com "não" não é reclamação |
| "São simplesmente perfeitos ❤❤❤" (Havaianas) | positivo | ✅ positivo | ❌ negativo | Contexto e emoji |
| "Fixava bem agr virou agua de chuchu." (O Boticário) | negativo | ❌ positivo | ❌ positivo | Expressão popular: "água de chuchu" |
| "Liz está surreal o preço" (O Boticário) | positivo | ❌ negativo | ❌ neutro | Gíria: "surreal" é elogio |
| "Anotem: O PS5 vai estar $2.800 a $3.200 com cupom!" (PS5) | neutro | ❌ negativo | ✅ neutro | Preço é informação, não queixa |
| "CALMA DAKI UNS DIAS VAO COMEÇAR A ESTRAGAR FIQUEM TRANKILOOS" (BYD) | negativo | ❌ neutro | ✅ negativo | Ironia e grafia fora do padrão |

As vantagens **comprovadas** do BERTimbau são de contexto: negação, perguntas e emojis. Ironia, gírias e expressões populares continuam erradas em todos os modelos. Não há evidência nos nossos dados de que o BERTimbau lide melhor com ironia, embora a atenção, em teoria, permita relacionar partes distantes da frase.

### 5.4 Separar os comentários sobre o produto (`05c`)

Usamos 174 comentários com gabarito: 86 sobre o produto e 88 não. Sem nenhum filtro, 49% são sobre o produto.

**NER como filtro.** A ideia era passar só os comentários com alguma marca, produto ou empresa. Não funcionou:
- o GLiNER2 acha uma entidade em ~85% dos comentários;
- dos que passam, só ~50% são sobre o produto, a mesma proporção de antes;
- o resultado foi o mesmo em todos os limiares (0,3 a 0,9).

O NER não está errando as entidades. A pergunta "tem uma entidade?" é que não equivale a "é opinião sobre o produto?": comentários do YouTube citam o youtuber, outras marcas e objetos o tempo todo.

**Busca pelo nome do produto e apelidos.**

| | Sobre o produto | Não é sobre o produto |
|---|---|---|
| **Cita o nome** | 31 | 3 |
| **Não cita** | 55 | 85 |

- **Precision = 31/34 = 0,91:** quase todo comentário que cita o nome é sobre o produto.
- **Recall = 31/86 = 0,36:** muitos comentários sobre o produto não citam o nome. Exemplo: "Esse interior novo […] é bem frágil e encarde, é tipo uma flanela" (BYD).

Por isso o app **não descarta** comentários. Ele separa os grupos e explica a diferença na tela.

### 5.5 Significância

Comparamos cada modelo com o TF-IDF nos 86 comentários, com bootstrap e McNemar:

| Modelo × TF-IDF | Diferença de F1 | IC 95% | Só o modelo acerta / só o TF-IDF acerta | p (McNemar) |
|---|---|---|---|---|
| BERTimbau congelado + RL | +0,064 | [−0,04; 0,17] | 12 / 7 | 0,36 |
| XLM-R tweets | +0,030 | [−0,07; 0,14] | 12 / 10 | 0,83 |
| BERTimbau ajustado (semente 42) | +0,018 | [−0,10; 0,14] | 11 / 8 | 0,65 |
| BiGRU + GloVe | −0,004 | [−0,12; 0,12] | 10 / 9 | 1,00 |

**Nenhuma diferença é significativa a 5%.** Com 86 comentários, o teste só detectaria diferenças muito grandes. Escolhemos o BERTimbau congelado pela **vantagem consistente** em todas as métricas do domínio real (F1, F1 do neutro, erro na proporção), e não por um teste significativo.

## 6. Discussão

**Comparação entre abordagens.**
- **Na loja,** a ordem segue a complexidade: TF-IDF < BiGRU < BERTimbau ajustado. A ressalva é que a rede neural só passa o TF-IDF com embeddings pré-treinados.
- **No YouTube,** essa ordem se desfaz. Os dois modelos que menos se especializam na loja ganham: o BERTimbau congelado, que só treina a camada final, e o XLM-R, que nunca viu a loja.
- **A lição** é que, sob troca de domínio, o desempenho no teste do domínio de origem não basta para escolher o modelo. É preciso uma validação, mesmo pequena, no domínio de uso.

**Erros encontrados.**
1. **Ironia, gírias e expressões populares** ("água de chuchu", "surreal"): todos os modelos erram.
2. **O neutro é raro no treino (5%)** e comum no YouTube (perguntas, informações de preço). O TF-IDF puxa perguntas com "não" para negativo.
3. **Relevância:** metade dos comentários não é sobre o produto, e nenhum modelo de sentimento foi treinado para reconhecer isso. Os modelos dão um sentimento a qualquer texto.
4. **NER:** confunde categorias e aceita termos genéricos ("17", "celular"), que aparecem na lista de marcas do app.

**Limitações técnicas.**
- **Troca de domínio:** o treino é todo de avaliações de loja de 2018. O estilo do YouTube (ironia, gírias, emojis, textos curtos) não aparece no treino.
- **Gabarito pequeno e enviesado para os casos fáceis:** 86 comentários com sentimento, formados só pelos itens de consenso. Os itens ambíguos, justamente os mais difíceis, ficaram fora, então as métricas no YouTube tendem a ser otimistas.
- **Sem significância estatística** entre os modelos no YouTube.
- **Dois anotadores, sem um terceiro para desempate,** e kappa moderado (0,48).
- **NER avaliado só em frases sintéticas** para P/R/F1 por entidade.
- **A separação pelo nome** perde 64% dos comentários sobre o produto (Recall 0,36).
- **A busca da API não é exaustiva:** devolve uma amostra por relevância. Produtos pequenos ou infantis podem não ter vídeos utilizáveis, como mostra o registro de falhas no README.

## 7. Aspectos éticos

**Privacidade.**
- O cliente da API descarta nome, foto e ID do canal do autor antes de salvar qualquer dado.
- Os textos coletados ficam só na máquina local, fora do git.
- No repositório vão apenas IDs de comentários e rótulos.
- Os exemplos deste relatório e da apresentação são trechos curtos, sem autor.

**Licenças e termos.**
- **RePro:** licença CC BY-NC-SA 4.0, só para uso não comercial. O classificador treinado nele, que está no repositório, segue a mesma licença.
- **YouTube:** os termos da API desaconselham redistribuir o conteúdo. Por isso os textos não vão para o GitHub.
- **Modelos pré-treinados:** o BERTimbau, o XLM-R e o GLiNER2 são baixados das fontes oficiais, e não copiados para o repositório.

**Vieses.**
- **Estilo:** o classificador aprendeu com consumidores que escreveram avaliações numa loja, um público e um estilo diferentes dos do YouTube.
- **Grafia:** erra mais em textos com ironia, gírias e grafia fora do padrão. Grupos que escrevem assim tendem a ser lidos com mais erro.
- **Rotulagem:** o gabarito reflete o julgamento de duas pessoas.

**Uso responsável.**
- **É um indicador:** o resultado mostra uma tendência, não uma medida exata da opinião do público. A amostra de vídeos é parcial, e há um erro de alguns pontos percentuais na proporção (seção 5.3).
- **Não serve para julgar pessoas:** o app não deve ser usado para identificar ou avaliar quem comenta.
- **Decisão humana:** decisões de negócio não devem se basear só na classificação automática. O app mostra os comentários originais justamente para que uma pessoa leia e confira.

## 8. Conclusão

**Resultados.** Entregamos um app funcional que, a partir de um produto e um período, coleta vídeos e comentários do YouTube, separa os que citam o produto, classifica o sentimento e mostra as marcas citadas.

O classificador escolhido, o BERTimbau congelado com Regressão Logística, chega a F1 0,807 nos comentários reais. Ele erra a proporção de sentimento em até 8% e não exige GPU para treinar.

**Principais achados.**
1. A troca de domínio é o problema central: todos os modelos perdem de 0,03 a 0,15 de F1, e o melhor na loja não é o melhor no YouTube.
2. Até para humanos a tarefa é mais difícil no YouTube (kappa 0,48 contra 0,71).
3. No YouTube, a relevância vem antes do sentimento: metade dos comentários não é sobre o produto. O NER acha entidades, mas não resolve a relevância. A busca pelo nome é precisa, mas incompleta.

**Possibilidades de evolução.**
- **Mais rótulos do YouTube:** rotular mais comentários, inclusive os ambíguos, para ajustar os modelos ao domínio e ter poder estatístico.
- **Classificador de relevância:** treinar um modelo para decidir se o comentário é sobre o produto, que capte as menções implícitas.
- **Categorias de intenção:** pergunta, elogio ao criador, reclamação e spam, no lugar do rótulo único "não é sobre o produto".
- **Transcrição do vídeo e chatbot com RAG:** a transcrição (por exemplo, com o Whisper) daria a opinião do criador. Junto com um chatbot com RAG sobre a transcrição e os comentários, permitiria responder às perguntas abertas.
- **Sentimento orientado a alvo:** distinguir o sentimento sobre o produto pesquisado do sentimento sobre os concorrentes citados.

## Referências

**Dados utilizados**
- SILVA, L. N. S. *et al.* RePro: A Benchmark Dataset for Opinion Mining in Brazilian Portuguese. *PROPOR*, 2024. Dados: <https://github.com/lucasnil/repro>.
- REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. *STIL*, 2019. Dados: <https://github.com/americanas-tech/b2w-reviews01>. Avaliada na proposta, mas não usada.

**Modelos e embeddings pré-treinados utilizados**
- BERTimbau: `neuralmind/bert-base-portuguese-cased`. SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: Pretrained BERT Models for Brazilian Portuguese. *BRACIS*, 2020. <https://huggingface.co/neuralmind/bert-base-portuguese-cased>.
- XLM-R de tweets: `cardiffnlp/twitter-xlm-roberta-base-sentiment`. BARBIERI, F.; ESPINOSA ANKE, L.; CAMACHO-COLLADOS, J. XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond. *LREC*, 2022. <https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment>.
- GLiNER2 PT-BR: `ottema/gliner2-ptbr`. <https://huggingface.co/ottema/gliner2-ptbr>.
- GloVe 300d do NILC-USP: `nilc-nlp/glove-300d`. HARTMANN, N. *et al.* Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks. *STIL*, 2017. <https://huggingface.co/nilc-nlp/glove-300d>.
- FastText para português (`cc.pt.300`). GRAVE, E. *et al.* Learning Word Vectors for 157 Languages. *LREC*, 2018. <https://fasttext.cc/docs/en/crawl-vectors.html>.

**Métodos**
- LANDIS, J. R.; KOCH, G. G. The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 1977. É a escala usual de interpretação do kappa, usada na seção 5.1.

**Ferramentas e documentação consultada**
- YouTube Data API v3, documentação oficial: <https://developers.google.com/youtube/v3>.
- scikit-learn, PyTorch, Hugging Face Transformers, gliner2, Streamlit e pandas: documentação oficial de cada biblioteca, nas versões fixadas em `src-pln/requirements.txt`.

**Material da disciplina**
- Material de aula de Processamento de Linguagem Natural, base da fundamentação teórica (seção 2) e do passo a passo do projeto.

## Anexo: onde está cada resultado

| Seção | Notebook | Arquivo de resultados |
|---|---|---|
| 5.2 | `03_reconhecimento_entidades` | (no notebook) |
| 5.3, rede neural | `04_rede_neural` | `04_resultados_rede_neural.csv` |
| 5.3, Transformers | `05_transformers` | `05_resultados_transformers.csv` |
| 5.1 e 5.3, YouTube | `05b_avaliacao_youtube` | `05b_resultados_youtube_todos.csv`, `05b_predicoes_youtube_todos.csv` |
| 5.4 e 5.5 | `05c_fluxo_do_app` | `05c_resultados_fluxo_app.csv`, `05c_significancia.csv` |
