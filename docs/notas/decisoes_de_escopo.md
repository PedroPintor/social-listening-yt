# Decisões de escopo

Desvios em relação ao passo a passo do professor, com a justificativa que vai para o relatório e para a apresentação.

## Etapa 6: aplicação em vez de chatbot

**O que o passo a passo pede.** Na etapa "Chatbot ou aplicação multimodal", construir uma interface de interação com o usuário. Ele dá como exemplo um chatbot que responde em linguagem natural a perguntas sobre os resultados, com respostas fundamentadas nos dados.

**O que fazemos.** Construímos uma aplicação: a empresa informa o produto e o período, o app coleta os vídeos, classifica os comentários e mostra o resumo, com o filtro por NER separando os comentários que citam o produto. Não há chatbot. O professor aceitou a troca, desde que justificada (conversa de novembro de 2026).

**Justificativa.**
- **As perguntas são sempre as mesmas.** Quem monitora uma marca quer saber quanto é positivo ou negativo, o que dizem os comentários que citam o produto e quais vídeos puxam a conversa. Um painel responde a tudo isso numa tela só; o chatbot obrigaria a pessoa a formular a pergunta.
- **O volume por consulta é pequeno.** Com um ou dois dias de vídeos, são algumas centenas de comentários. Gráficos e filtros dão conta; não há o que "pesquisar" em linguagem natural.
- **Risco sem ganho.** Um chatbot exigiria um LLM com RAG, que pode inventar respostas, custa por chamada e precisaria de uma avaliação própria. Isso é outro projeto e foge do foco do trabalho, que é sentimento e NER.
- **O requisito de interface continua atendido.** O app é a interface com o usuário e é onde os modelos (classificador de sentimento e NER) se integram.

**Quando faria sentido (trabalho futuro).** Com muitos produtos ou períodos longos, perguntas abertas sobre os resultados ("do que mais reclamam do iPhone desde o lançamento?") passariam a valer um chatbot com RAG sobre os comentários já classificados. Ver também o critério definido pela empresa via LLM em [limitacoes_e_trabalhos_futuros.md](limitacoes_e_trabalhos_futuros.md).
