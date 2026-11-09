# YouTube Data API: o que precisamos saber neste projeto

## 1. Chave (gratuita, sem cartão)

1. Entre em <https://console.cloud.google.com/> com uma conta Google e crie um projeto.
2. Em **APIs e serviços → Biblioteca**, ative a **YouTube Data API v3**.
3. Em **APIs e serviços → Credenciais → Criar credenciais → Chave de API**. Se quiser, restrinja a chave à YouTube Data API v3.
4. Na raiz do repositório, copie `.env.example` para `.env` e preencha `YOUTUBE_API_KEY=...`.
   O `.env` está no `.gitignore`: nunca commite a chave nem a cole em notebooks.

Não é preciso ativar faturamento nem cadastrar cartão.

## 2. Cota

- **10.000 unidades por dia**, por projeto, de graça. Não há cobrança: ao estourar, a API responde
  `403 quotaExceeded` até a cota zerar (meia-noite no horário do Pacífico, por volta das 4h/5h de Brasília).
- Custos que usamos:

| Chamada | Custo | Rende |
|---|---|---|
| `search.list` (buscar vídeos) | **100** | até 50 vídeos |
| `videos.list` (estatísticas, `madeForKids`) | 1 | até 50 vídeos |
| `commentThreads.list` (comentários) | 1 | até 100 comentários |

- **Coleta da amostra** (`gerar_amostra.py`, 6 produtos): ~650 unidades (máximo 756).
- **Uma consulta do app** (1 produto, 1 página de busca, ~20 vídeos com comentários): ~120 unidades,
  ou seja, ~80 consultas por dia. A busca é o que pesa; os comentários são baratos.
- `youtube.py` guarda cada resposta em cache no disco (repetir não gasta) e conta as unidades gastas por execução.

## 3. Decisão: o período vale pela data do vídeo

O período informado pela empresa filtra a **data de publicação do vídeo** (`publishedAfter`/`publishedBefore`).
Todos os comentários desses vídeos entram, sejam de quando forem.

## 4. Limitações que afetam o projeto

- **Região = país.** Usamos `regionCode=BR` e `relevanceLanguage=pt`. A API não informa de onde é quem comenta;
  não dá para filtrar por estado ou cidade. Comentários em outros idiomas são descartados por um detector simples.
- **Vídeos infantis não têm comentários.** Vídeos `madeForKids` são pulados; outros podem ter comentários
  desativados (`commentsDisabled`), que tratamos como vídeo sem comentários.
- **A busca não é exaustiva.** O `search.list` devolve uma amostra por relevância, não todos os vídeos do período.
  Os números do app são indicativos, não um censo.
- **Sem dislikes.** A API não expõe mais a contagem de "não gostei".
- **Dados brutos fora do git.** Os termos da API desaconselham redistribuir o conteúdo: tudo o que é coletado fica em
  `src-pln/coleta_youtube/dados/`, ignorada pelo git. No relatório, só estatísticas e exemplos curtos.
- **Anonimização.** Nome, canal e ID de quem comenta são descartados na coleta, antes de salvar qualquer coisa.
