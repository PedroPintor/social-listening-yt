# Limitações e trabalhos futuros

Observações feitas durante a rotulagem manual dos comentários e títulos do YouTube (novembro de 2026). Servem de base para as seções de Discussão e Conclusão do relatório e de roteiro para a continuação do projeto.

> Os números marcados com **[a preencher]** entram quando a rotulagem terminar.

## 1. Sentimento do título depende de quem olha

**Problema.** Classificar o título de um vídeo como positivo, negativo ou neutro não faz sentido sem um ponto de vista. "iPhone 17 vs Galaxy S25: qual comprar?" é neutro para quem rotula, mas pode ser negativo para a Apple se o vídeo concluir a favor da Samsung.

**Conceitos.**
- *Sentimento orientado a alvo* e *detecção de posicionamento* (*stance detection*): a pergunta certa não é "o texto é positivo?", e sim "o texto é a favor ou contra X, do ponto de vista de Y?".
- Títulos são **informativos por gênero**: eles anunciam o tipo de conteúdo ("unboxing", "review", "vale a pena?") e quase não carregam opinião.

**Evidência.** [a preencher] % dos títulos rotulados como neutros.

**Decisão no projeto atual.** O app não atribui sentimento ao título, ou o mostra apenas como informação, deixando explícita essa limitação. O sentimento vem dos comentários.

## 2. O vídeo tem três camadas de opinião

| Camada | Pergunta | Fonte | Situação |
|---|---|---|---|
| Vídeo | Que tipo de conteúdo é? | título (e metadados) | trabalho futuro |
| Criador | O youtuber falou bem ou mal do produto? | conteúdo falado (transcrição) | trabalho futuro |
| Público | O que as pessoas acharam? | comentários | **feito no projeto** |

**Ideias futuras.**
- **Categoria do vídeo** (unboxing, review, comparativo, reação, tutorial, reclamação): classificação de gênero de conteúdo, com regras por palavra-chave no título ou com um classificador. O `categoryId` da API existe, mas é escolhido pelo próprio criador e é genérico demais ("Ciência e tecnologia", "Pessoas e blogs").
- **Opinião do criador:** exige a transcrição do áudio (*speech-to-text*, por exemplo com o Whisper). A API não fornece legendas de vídeos de terceiros, e baixar o áudio esbarra nos termos de uso.
- **Critério definido pela empresa:** a empresa descreve o que é positivo ou negativo para ela, e um LLM classifica com esse critério no prompt (classificação *zero-shot* condicionada ao usuário).

## 3. A maioria dos comentários não é sobre o produto

**Problema.** Avaliações de loja (RePro, usado no treino) falam do produto por construção. No YouTube, a conversa é aberta: elogios ao youtuber, piadas, pedidos de vídeo, perguntas, spam. A troca de domínio é maior do que o previsto, porque muda não só o *jeito* de escrever, mas também **sobre o que** se escreve.

**Conceitos.**
- **Filtragem de relevância:** em *social listening*, é a etapa que vem antes do sentimento.
- **Classificação de intenção** (ou atos de fala): pergunta sobre o produto, comentário sobre o criador ou o vídeo, pedido, piada, spam.

**Evidência.** [a preencher] % dos comentários rotulados como "não é sobre o produto".

**Impacto na avaliação.** A avaliação no YouTube é feita em duas partes:
1. A proporção de ruído ("não é sobre o produto").
2. As métricas de sentimento só nos comentários sobre o produto, que é a comparação justa entre os modelos, já que nenhum deles viu ruído no treino. Também medimos o que cada modelo faz com o ruído.

**Ideia futura.** Substituir o rótulo único "não é sobre o produto" por categorias de intenção.

## 4. O NER como filtro de menções, e seu limite

**Uso no projeto.** O NER (GLiNER2) separa os comentários em dois grupos: os que **citam o produto ou a marca explicitamente** e os demais. Assim, quem usa o app vê primeiro o que fala diretamente do produto. Dá para avaliar esse filtro com a própria rotulagem, comparando "cita o produto" com o rótulo "não é sobre o produto" (precisão e revocação do filtro).

**Limite.** Muitos comentários relevantes não citam o nome do produto ("a bateria dura quanto?", "comprei ontem e chegou riscado"). O filtro por NER deixa esses comentários de fora, então "não cita" não significa "não é sobre o produto".

**Ideia futura.** Um classificador de relevância treinado com os rótulos do YouTube, que capte as menções implícitas.
