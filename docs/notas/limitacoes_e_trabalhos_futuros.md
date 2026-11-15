# Limitações e trabalhos futuros

Observações feitas durante a rotulagem manual dos comentários e títulos do YouTube (novembro de 2026). Servem de base para as seções de Discussão e Conclusão do relatório e de roteiro para a continuação do projeto.

Amostra rotulada: 350 itens (300 comentários e 50 títulos), rotulados de forma independente por Pedro e Maria Alice.

## 1. Sentimento do título depende de quem olha

**Problema.** Classificar o título de um vídeo como positivo, negativo ou neutro não faz sentido sem um ponto de vista. "iPhone 17 vs Galaxy S25: qual comprar?" é neutro para quem rotula, mas pode ser negativo para a Apple se o vídeo concluir a favor da Samsung.

**Conceitos.**
- *Sentimento orientado a alvo* e *detecção de posicionamento* (*stance detection*): a pergunta certa não é "o texto é positivo?", e sim "o texto é a favor ou contra X, do ponto de vista de Y?".
- Títulos são **informativos por gênero**: eles anunciam o tipo de conteúdo ("unboxing", "review", "vale a pena?") e quase não carregam opinião.

**Evidência.** Dos 50 títulos, 92% foram rotulados como neutros pelo Pedro e 74% pela Maria Alice. O kappa nos títulos foi de apenas 0,36: sem um ponto de vista definido, nem os rotuladores concordam.

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

**Evidência.** Dos 300 comentários, 51% foram rotulados como "não é sobre o produto" pelo Pedro e 36% pela Maria Alice. É também o ponto de maior divergência: o kappa geral (5 classes) foi 0,48 (moderado), mas sobe para 0,63 (substancial) quando se olha só positivo, neutro e negativo nos itens que ambos consideraram sobre o produto. Ou seja, decidir **se** o comentário é sobre o produto é mais difícil do que decidir a polaridade. Exemplo típico: elogio ao youtuber ("ótimo vídeo") conta como positivo ou como fora do produto?

**Impacto na avaliação.** A avaliação no YouTube é feita em duas partes:
1. A proporção de ruído ("não é sobre o produto").
2. As métricas de sentimento só nos comentários sobre o produto, que é a comparação justa entre os modelos, já que nenhum deles viu ruído no treino. Também medimos o que cada modelo faz com o ruído.

**Ideia futura.** Substituir o rótulo único "não é sobre o produto" por categorias de intenção.

## 4. Separar os comentários sobre o produto: NER e busca pelo nome

**Hipótese testada.** Usar o NER (GLiNER2) como filtro, passando só os comentários que citam uma marca, produto ou empresa. A avaliação comparou o filtro com o rótulo "não é sobre o produto" (precisão e revocação).

**Uso no projeto.** Não funcionou (evidência abaixo). No app, a separação é feita pela **busca do nome do produto** e de apelidos, sem descartar comentários. O NER só **lista as marcas e produtos citados**, por exemplo os concorrentes. Ver [decisoes_de_escopo.md](decisoes_de_escopo.md).

**Limite.** Muitos comentários relevantes não citam o nome do produto ("a bateria dura quanto?", "comprei ontem e chegou riscado"). Então "não cita" não significa "não é sobre o produto".

**Evidência** (notebook `05c_fluxo_do_app.ipynb`, 174 comentários com gabarito: os 86 sobre o produto, base das métricas de sentimento, mais os 88 "não é sobre o produto"):
- **O NER não filtra.** O GLiNER2 acha alguma marca, produto ou empresa em ~85% dos comentários, e só ~50% dos que passam são sobre o produto, a mesma proporção de não filtrar. Isso vale para todos os limiares e categorias testados. Comentários do YouTube citam outras marcas, o youtuber e objetos comuns o tempo todo.
- **Buscar o nome do produto é preciso, mas perde muito.** 91% dos comentários que citam o nome (ou apelidos) são sobre o produto, mas só 36% dos comentários sobre o produto citam o nome.

**Ideia futura.** Um classificador de relevância treinado com os rótulos do YouTube, que capte as menções implícitas.
