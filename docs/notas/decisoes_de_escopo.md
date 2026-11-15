# Decisões de escopo

Desvios em relação ao passo a passo do professor, com a justificativa que vai para o relatório e para a apresentação.

## Etapa 6: aplicação em vez de chatbot

**O que o passo a passo pede.** Na etapa "Chatbot ou aplicação multimodal", construir uma interface de interação com o usuário. Ele dá como exemplo um chatbot que responde em linguagem natural a perguntas sobre os resultados, com respostas fundamentadas nos dados.

**O que fazemos.** Construímos uma aplicação: a empresa informa o produto e o período, o app coleta os vídeos, classifica os comentários e mostra o resumo, com o filtro por NER separando os comentários que citam o produto. Não há chatbot. O professor aceitou a troca, desde que justificada (conversa de novembro de 2026).

**Justificativa.**
- **As perguntas são sempre as mesmas.** Quem monitora uma marca quer saber quanto é positivo ou negativo, o que dizem os comentários que citam o produto e quais vídeos puxam a conversa. Um painel responde a tudo isso numa tela só; o chatbot obrigaria a pessoa a formular a pergunta.
- **O volume por consulta é pequeno.** Com um ou dois dias de vídeos, são algumas centenas de comentários. Gráficos e filtros dão conta; não há o que "pesquisar" em linguagem natural.
- **O chatbot só agregaria com um contexto que não temos.** Ele valeria a pena se o usuário pudesse perguntar sobre o *vídeo* ("o que o youtuber achou da bateria?") e sobre o *conjunto* dos comentários ao mesmo tempo. Para isso, seria preciso:
  - o conteúdo falado do vídeo, que a API do YouTube não fornece para vídeos de terceiros. Obtê-lo exigiria outra fonte: baixar o áudio e transcrever (por exemplo com o Whisper), o que esbarra nos termos de uso;
  - um LLM com RAG sobre a transcrição e os comentários, para que as respostas se apoiem nos dados e não sejam inventadas. Isso custa por chamada e precisaria de uma avaliação própria.

  Sem esse contexto, o chatbot só repetiria o que o painel já mostra. Com ele, o escopo cresceria para outro projeto e fugiria do foco do trabalho, que é sentimento e NER.
- **O requisito de interface continua atendido.** O app é a interface com o usuário e é onde os modelos (classificador de sentimento e NER) se integram.

**Quando faria sentido (trabalho futuro).** Quando houver a transcrição dos vídeos (a camada "criador", na nota de limitações), ou com muitos produtos e períodos longos, perguntas abertas ("do que mais reclamam do iPhone desde o lançamento?") passariam a valer um chatbot com RAG sobre a transcrição e os comentários já classificados. Ver também o critério definido pela empresa via LLM em [limitacoes_e_trabalhos_futuros.md](limitacoes_e_trabalhos_futuros.md).

## Classificador do app: BERTimbau congelado + Regressão Logística

**Evidência** (notebooks `05b` e `05c`, comentários do YouTube com gabarito):

| Modelo | F1 RePro | F1 YouTube (86) | F1 neutro YouTube | Erro no resumo* | CPU, 300 comentários |
|---|---|---|---|---|---|
| TF-IDF + RL | 0,883 | 0,744 | 0,667 | 16 pp | 0,1 s |
| BiGRU + GloVe ajustado | 0,892 | 0,739 | 0,595 | 11 pp | ~3 s |
| XLM-R tweets (zero-shot) | 0,805 | 0,774 | 0,667 | 6 pp | ~60 s |
| **BERTimbau congelado + RL** | 0,886 | **0,807** | **0,727** | 8 pp | ~50 s |
| BERTimbau ajustado | **0,921** | 0,788 ± 0,031 | 0,671 | 11 pp | ~50 s |

\* Maior diferença, em pontos percentuais, entre a proporção de positivo/neutro/negativo prevista e a real, nos comentários sobre o produto.

**Por que ele.** É o melhor no domínio real (YouTube) em F1 macro e no neutro, a classe mais difícil, e o resumo que ele mostra fica perto do real. Ele não especializa no estilo do RePro, porque só a Regressão Logística é treinada. O custo (~50 s para algumas centenas de comentários, ~450 MB de modelo) é aceitável para uma análise feita sob demanda, que já espera a coleta da API.

**Ressalva honesta.** Com 86 comentários, nenhuma diferença contra o TF-IDF é estatisticamente significativa (bootstrap e McNemar, `05c`). A escolha se apoia na vantagem consistente em todas as métricas do domínio real, não em um teste significativo. Confirmar com mais dados rotulados do YouTube é trabalho futuro.

**Por que não os outros.**
- **TF-IDF:** puxa para negativo (diria 53% negativo quando o real é 37%) e manda 59% do ruído para negativo. Fica como linha de base.
- **BiGRU:** custo baixo, mas é a pior no YouTube (F1 0,739) e no neutro (0,595).
- **BERTimbau ajustado:** o melhor no RePro, mas cai mais no YouTube: especializou no estilo das avaliações de loja. Varia mais entre sementes e exige GPU para treinar.
- **XLM-R zero-shot:** bom resultado sem treino, mas é o maior e o mais lento (278 milhões de parâmetros, ~1,1 GB) e não usa nenhum dado nosso.

## NER no app: mostrar as marcas citadas, não filtrar

**O que testamos.** Usar o NER como filtro de relevância: passa o comentário em que o GLiNER2 acha uma marca, produto ou empresa (`05c`).

**Resultado.** O filtro não separa: a proporção de ruído é a mesma antes (51%) e depois (50%) dele, em todos os limiares e categorias. O NER não está errando as entidades; a pergunta "o comentário tem uma entidade?" é que não equivale a "é opinião sobre o produto?". Comentários do YouTube citam o youtuber, outras marcas e objetos o tempo todo.

**Como o app usa.**
- **Separação dos comentários:** busca do nome do produto e apelidos (91% de precisão). Nenhum comentário é descartado: o app mostra o grupo "citam o produto" (base do resumo principal) e o grupo "demais comentários", com o aviso de que muitos não são sobre o produto.
- **NER:** lista as marcas e produtos citados nos comentários, por exemplo os concorrentes. Isso responde a uma pergunta real de monitoramento de marca ("com quem me comparam?"), algo que a busca pelo nome não faz, porque não sabe de antemão quais marcas procurar.

**Trabalho futuro.** Um classificador de relevância treinado com os rótulos do YouTube.
