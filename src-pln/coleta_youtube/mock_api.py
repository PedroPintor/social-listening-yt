"""Transporte simulado da YouTube Data API (sem rede, sem cota). Usado nos testes e em --mock.

Imita o formato real das respostas de search.list, videos.list e commentThreads.list,
inclusive com campos de autor (para testar a anonimização), um vídeo infantil,
um vídeo com comentários desativados, spam, links e comentários em outros idiomas.
"""

from __future__ import annotations

import random

FRASES_PT = [
    "Comprei o {p} semana passada e estou gostando muito, valeu cada centavo",
    "Achei o {p} caro demais pelo que entrega, não recomendo",
    "Alguém sabe se o {p} tem garantia no Brasil?",
    "O {p} é bom mas a bateria podia durar mais",
    "Melhor compra que fiz esse ano, o {p} é sensacional",
    "Que vídeo bom, explicou tudo sobre o {p} sem enrolação",
    "Meu {p} veio com defeito e a assistência não resolve nada",
    "Ah claro, o {p} é perfeito, só travou três vezes hoje kkkk",
    "To em dúvida entre o {p} e o concorrente, qual vocês acham melhor?",
    "Esse canal é muito bom, sempre assisto os vídeos",
    "Gostei do design do {p} mas o preço tá absurdo",
    "Uso o {p} faz um mês e não tenho nada pra reclamar",
    "O atendimento da loja foi péssimo mas o {p} em si é ótimo",
    "Vale a pena esperar baixar o preço do {p}",
    "Primeira vez que vejo um review honesto do {p}, parabéns",
]
FRASES_OUTRAS = [
    "Este {p} es muy bueno pero muy caro",           # espanhol
    "I love my new {p}, best purchase ever",         # inglês
    "primeiro!",                                      # curto
    "Ganhe dinheiro em casa, acesse https://golpe.exemplo.com",  # link/spam
    "Se inscreve no meu canal que eu sigo de volta",  # spam
]


def criar_transporte(semente: int = 0, quota_apos: int | None = None):
    """Devolve transporte(endpoint, params) -> (status, json).

    quota_apos: se definido, a partir dessa chamada responde 403 quotaExceeded.
    """
    estado = {"n": 0}

    def transporte(endpoint: str, params: dict):
        estado["n"] += 1
        if quota_apos is not None and estado["n"] > quota_apos:
            return 403, {"error": {"code": 403, "message": "quota", "errors": [{"reason": "quotaExceeded"}]}}
        rng = random.Random(f"{semente}|{endpoint}|{sorted(params.items())}")

        if endpoint == "search":
            q = params["q"]
            n = min(int(params.get("maxResults", 5)), 12)
            itens = [{"id": {"kind": "youtube#video", "videoId": f"{q[:4].replace(' ', '_')}{i:02d}"},
                      "snippet": {"title": f"Review do {q} parte {i}" if i % 3 else f"{q}: vale a pena em 2026?",
                                  "channelTitle": f"Canal {i}", "publishedAt": "2026-11-08T12:00:00Z"}}
                     for i in range(n)]
            return 200, {"items": itens}

        if endpoint == "videos":
            itens = []
            for vid in params["id"].split(","):
                i = int(vid[-2:])
                itens.append({"id": vid,
                              "snippet": {"title": f"Review do {vid[:-2]} parte {i}", "channelTitle": f"Canal {i}",
                                          "publishedAt": "2026-11-08T12:00:00Z", "defaultAudioLanguage": "pt-BR"},
                              "statistics": {"viewCount": str(rng.randint(100, 90000)),
                                             "likeCount": str(rng.randint(5, 5000)),
                                             "commentCount": str(0 if i == 11 else rng.randint(10, 800))},
                              "status": {"madeForKids": i == 7}})
            return 200, {"items": itens}

        if endpoint == "commentThreads":
            vid = params["videoId"]
            if vid.endswith("05"):
                return 403, {"error": {"code": 403, "message": "disabled", "errors": [{"reason": "commentsDisabled"}]}}
            pagina = int(params.get("pageToken", "0"))
            n = int(params.get("maxResults", 20))
            produto = vid[:-2].replace("_", " ")
            itens = []
            for k in range(n):
                frase = rng.choice(FRASES_PT if rng.random() < 0.8 else FRASES_OUTRAS)
                texto = frase.format(p=produto) + ("" if rng.random() < 0.7 else f" ({vid}-{pagina}-{k})")
                itens.append({"id": f"Ug{vid}{pagina}{k:03d}", "snippet": {"totalReplyCount": rng.randint(0, 3), "topLevelComment": {"snippet": {
                    "textOriginal": texto, "textDisplay": texto, "likeCount": rng.randint(0, 300),
                    "publishedAt": "2026-11-08T15:00:00Z",
                    "authorDisplayName": "@pessoa_real", "authorChannelId": {"value": "UCxxxx"},
                    "authorProfileImageUrl": "https://exemplo/foto.jpg", "authorChannelUrl": "https://exemplo"}}}})
            resp = {"items": itens}
            if pagina < 1:
                resp["nextPageToken"] = str(pagina + 1)
            return 200, resp

        return 404, {"error": {"errors": [{"reason": "notFound"}]}}

    return transporte
