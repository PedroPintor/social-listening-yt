# Coleta do YouTube e rotulagem manual

Amostra de ~300 comentários e ~50 títulos reais do YouTube, rotulada à mão pelos dois integrantes,
para **testar** quanto os modelos treinados no RePro (Americanas) perdem ao mudar de domínio.
Detalhes da API (chave, cota, limitações): [`docs/notas/youtube_api.md`](../../docs/notas/youtube_api.md).

| Arquivo | Para quê |
|---|---|
| `youtube.py` | Cliente da API (busca, detalhes, comentários, cache, contador de cota). Base do app da etapa 6. |
| `gerar_amostra.py` | Coleta os 6 produtos e sorteia a amostra estratificada. Produtos e período no topo do arquivo. |
| `rotulador.html` | Ferramenta de rotulagem (abre direto no navegador, sem servidor). |
| `kappa.py` | Kappa de Cohen, matriz de concordância e lista de divergências. |
| `mock_api.py`, `teste_mock.py` | API simulada e testes (não gastam cota). |
| `dados/` | Cache, brutos, amostra e rótulos de cada pessoa. **Ignorada pelo git** (não redistribuir textos). |
| `rotulos/rotulos_consolidados.csv` | Gerado pelo `kappa.py`: ids e rótulos, **sem texto**. Este vai para o git. |

Produtos da amostra (09/11/2026, vídeos de 02/11 a 08/11): iPhone 17, iFood, O Boticário (perfumes),
PlayStation 5, BYD Dolphin Mini e Havaianas. AirPods Pro 3, Air fryer Mondial e Gelelé foram testados e
trocados por falta de comentários em português (o histórico está no topo de `gerar_amostra.py`).

## Ordem de uso

Rode os comandos dentro desta pasta, com o `.venv` da raiz ativo.

1. **Chave:** copie `.env.example` (na raiz) para `.env` e preencha `YOUTUBE_API_KEY`.
2. **Testes (sem cota):** `python teste_mock.py`
3. **Ver o plano:** `python gerar_amostra.py` mostra as buscas, o período e a estimativa de unidades, sem chamar a API.
4. **Coletar:** `python gerar_amostra.py --executar` gera `dados/brutos/` e `dados/amostra_rotulagem.csv`
   (~650 unidades das 10.000 diárias). Rodar de novo reaproveita o cache. Para só refazer o sorteio: `--so-amostrar`.
5. **Rotular (cada um sozinho):** abra `rotulador.html`, carregue `dados/amostra_rotulagem.csv`, digite seu nome,
   rotule com as teclas 1 a 5 e clique em *Exportar CSV*. Salve o arquivo como `dados/rotulos_<nome>.csv`.
   O progresso fica salvo no navegador; não rotulem juntos nem comparem antes de exportar.
   A outra pessoa recebe o `amostra_rotulagem.csv` por fora do git (WhatsApp, Drive) e devolve o seu `rotulos_<nome>.csv`.
6. **Concordância:** `python kappa.py dados/rotulos_pedro.csv dados/rotulos_maria.csv`.
   Resolvam juntos as linhas de `dados/divergencias.csv` (coluna `rotulo_final`) e rodem de novo: o que já foi
   resolvido é preservado, e `rotulos/rotulos_consolidados.csv` (sem texto) fica pronto para o commit.

Rótulos: `positivo`, `neutro`, `negativo`, `misto`, `nao_sobre_produto` (o guia está no próprio rotulador).

Dependências: `requests`, `python-dotenv`, `pandas`, `scikit-learn`.
