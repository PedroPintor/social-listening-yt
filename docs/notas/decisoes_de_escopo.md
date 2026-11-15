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
