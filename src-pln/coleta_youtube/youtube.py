"""Cliente mínimo da YouTube Data API v3 para o projeto de monitoramento de marcas.

Reutilizável: é a base da coleta da amostra (gerar_amostra.py) e do app da etapa 6.

- A chave vem do `.env` na raiz do repositório (YOUTUBE_API_KEY); nunca é impressa.
- Toda resposta é anonimizada (campos `author*` removidos) ANTES de ir para o cache.
- Cache em disco por requisição: repetir a mesma chamada não gasta cota.
- `cliente.unidades` conta as unidades gastas nesta execução (só chamadas reais).
- `commentsDisabled` vira lista vazia; `quotaExceeded` liga `cliente.cota_esgotada`
  e as funções devolvem o que já tinham coletado, sem levantar exceção.

Uso:
    from youtube import YouTube
    yt = YouTube()
    videos = yt.buscar_videos("iPhone 17", "2026-11-07T00:00:00Z", "2026-11-09T00:00:00Z")
    videos = yt.detalhes_videos([v["video_id"] for v in videos])   # já sem os infantis
    comentarios = yt.comentarios(videos[0]["video_id"], limite=100)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import requests

URL_BASE = "https://www.googleapis.com/youtube/v3/"
CUSTO = {"search": 100, "videos": 1, "commentThreads": 1}  # unidades por chamada

PASTA = Path(__file__).resolve().parent
RAIZ_REPO = PASTA.parents[1]
CACHE_PADRAO = PASTA / "dados" / "cache"

# transporte(endpoint, params) -> (status_http, json). Trocado por um mock nos testes.
Transporte = Callable[[str, dict], tuple[int, dict]]


class CotaEsgotada(Exception):
    """A cota diária (10.000 unidades) acabou; zera por volta das 4h/5h de Brasília."""


class ErroAPI(Exception):
    def __init__(self, status: int, motivo: str, mensagem: str = ""):
        super().__init__(f"HTTP {status} ({motivo}) {mensagem}".strip())
        self.status, self.motivo = status, motivo


def carregar_chave() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ_REPO / ".env")
    except ImportError:  # python-dotenv ausente: depende da variável de ambiente
        pass
    chave = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not chave:
        raise RuntimeError("YOUTUBE_API_KEY não encontrada. Copie .env.example para .env e preencha.")
    return chave


def _transporte_http(chave: str) -> Transporte:
    def chamar(endpoint: str, params: dict) -> tuple[int, dict]:
        r = requests.get(URL_BASE + endpoint, params={**params, "key": chave}, timeout=30)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {"errors": [{"reason": "respostaInvalida"}], "message": r.text[:200]}}
    return chamar


def _anonimizar(obj):
    """Remove recursivamente authorDisplayName, authorChannelId, authorProfileImageUrl etc."""
    if isinstance(obj, dict):
        return {k: _anonimizar(v) for k, v in obj.items() if not k.startswith("author")}
    if isinstance(obj, list):
        return [_anonimizar(v) for v in obj]
    return obj


def _motivo(resposta: dict) -> tuple[str, str]:
    erro = resposta.get("error", {})
    erros = erro.get("errors") or [{}]
    return erros[0].get("reason", "desconhecido"), erro.get("message", "")


class YouTube:
    def __init__(self, chave: str | None = None, pasta_cache: Path | str | None = CACHE_PADRAO,
                 transporte: Transporte | None = None, verbose: bool = False):
        if transporte is None:
            transporte = _transporte_http(chave or carregar_chave())
        self._chamar = transporte
        self.pasta_cache = Path(pasta_cache) if pasta_cache else None
        self.unidades = 0          # gastas nesta execução (cache não conta)
        self.chamadas_reais = 0
        self.acertos_cache = 0
        self.cota_esgotada = False
        self.verbose = verbose

    # ------------------------------------------------------------------ núcleo
    def _arquivo_cache(self, endpoint: str, params: dict) -> Path | None:
        if not self.pasta_cache:
            return None
        chave = json.dumps({"e": endpoint, "p": params}, sort_keys=True, ensure_ascii=False)
        return self.pasta_cache / endpoint / (hashlib.sha256(chave.encode()).hexdigest()[:24] + ".json")

    def _get(self, endpoint: str, params: dict) -> dict:
        arq = self._arquivo_cache(endpoint, params)
        if arq and arq.exists():
            self.acertos_cache += 1
            return json.loads(arq.read_text(encoding="utf-8"))
        if self.cota_esgotada:
            raise CotaEsgotada()

        status, resposta = self._chamar(endpoint, params)
        self.chamadas_reais += 1
        self.unidades += CUSTO.get(endpoint, 1)  # chamadas com erro também consomem cota
        if self.verbose:
            print(f"  [API] {endpoint} ({CUSTO.get(endpoint, 1)} un.) total={self.unidades}")

        if status != 200 or "error" in resposta:
            motivo, msg = _motivo(resposta)
            if motivo in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
                self.cota_esgotada = True
                raise CotaEsgotada(msg)
            raise ErroAPI(status, motivo, msg)

        resposta = _anonimizar(resposta)
        if arq:
            arq.parent.mkdir(parents=True, exist_ok=True)
            arq.write_text(json.dumps(resposta, ensure_ascii=False), encoding="utf-8")
        return resposta

    # ------------------------------------------------------------------ busca
    def buscar_videos(self, consulta: str, publicado_depois: str, publicado_antes: str,
                      max_videos: int = 50, ordem: str = "relevance",
                      regiao: str = "BR", idioma: str = "pt") -> list[dict]:
        """search.list (100 un. por página de até 50). Datas em RFC 3339, ex.: 2026-11-07T00:00:00Z.

        O período filtra pela DATA DE PUBLICAÇÃO DO VÍDEO. A busca não é exaustiva.
        """
        videos, token = [], None
        while len(videos) < max_videos:
            params = {"part": "snippet", "q": consulta, "type": "video", "order": ordem,
                      "regionCode": regiao, "relevanceLanguage": idioma,
                      "publishedAfter": publicado_depois, "publishedBefore": publicado_antes,
                      "maxResults": min(50, max_videos - len(videos))}
            if token:
                params["pageToken"] = token
            try:
                r = self._get("search", params)
            except CotaEsgotada:
                break
            for item in r.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    sn = item.get("snippet", {})
                    videos.append({"video_id": vid, "titulo": sn.get("title", ""),
                                   "canal": sn.get("channelTitle", ""),
                                   "data_video": sn.get("publishedAt", "")})
            token = r.get("nextPageToken")
            if not token:
                break
        return videos

    def detalhes_videos(self, ids: list[str], pular_infantis: bool = True) -> list[dict]:
        """videos.list (1 un. por lote de 50): título, estatísticas e status.madeForKids."""
        saida = []
        for i in range(0, len(ids), 50):
            lote = ids[i:i + 50]
            try:
                r = self._get("videos", {"part": "snippet,statistics,status", "id": ",".join(lote),
                                         "maxResults": 50})
            except CotaEsgotada:
                break
            for item in r.get("items", []):
                st, sn, es = item.get("statistics", {}), item.get("snippet", {}), item.get("status", {})
                infantil = bool(es.get("madeForKids", False))
                if infantil and pular_infantis:
                    continue
                saida.append({
                    "video_id": item["id"], "titulo": sn.get("title", ""),
                    "canal": sn.get("channelTitle", ""), "data_video": sn.get("publishedAt", ""),
                    "idioma_audio": sn.get("defaultAudioLanguage", ""),
                    "views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0)),
                    "n_comentarios": int(st.get("commentCount", 0)) if "commentCount" in st else None,
                    "infantil": infantil,
                })
        return saida

    def comentarios(self, video_id: str, limite: int = 100, ordem: str = "relevance",
                    termos: str | None = None) -> list[dict]:
        """commentThreads.list (1 un. por página de até 100). Só comentários de primeiro nível.

        `termos` usa o parâmetro searchTerms (filtra comentários que contêm o texto).
        Vídeos com comentários desativados devolvem [] (sem quebrar).
        """
        saida, token = [], None
        while len(saida) < limite:
            params = {"part": "snippet", "videoId": video_id, "textFormat": "plainText",
                      "order": ordem, "maxResults": min(100, limite - len(saida))}
            if termos:
                params["searchTerms"] = termos
            if token:
                params["pageToken"] = token
            try:
                r = self._get("commentThreads", params)
            except CotaEsgotada:
                break
            except ErroAPI as e:
                if e.motivo in ("commentsDisabled", "videoNotFound", "forbidden"):
                    break
                raise
            for item in r.get("items", []):
                sn = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                saida.append({"video_id": video_id, "comment_id": item.get("id", ""), "texto": sn.get("textOriginal") or sn.get("textDisplay", ""),
                              "likes": int(sn.get("likeCount", 0)), "data_comentario": sn.get("publishedAt", ""),
                              "n_respostas": int(item.get("snippet", {}).get("totalReplyCount", 0))})
            token = r.get("nextPageToken")
            if not token:
                break
        return saida[:limite]

    def resumo(self) -> str:
        return (f"{self.unidades} unidades gastas em {self.chamadas_reais} chamadas reais; "
                f"{self.acertos_cache} vindas do cache" + (" — COTA ESGOTADA" if self.cota_esgotada else ""))


def estimar_unidades(n_buscas: int, paginas_busca: int, n_videos: int, paginas_comentarios: int) -> int:
    """Estimativa pessimista (sem cache): buscas + detalhes + comentários."""
    return (n_buscas * paginas_busca * CUSTO["search"]
            + -(-n_videos // 50) * CUSTO["videos"]
            + n_videos * paginas_comentarios * CUSTO["commentThreads"])
