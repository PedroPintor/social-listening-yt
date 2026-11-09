"""Coleta comentários e títulos do YouTube e sorteia a amostra para rotulagem manual.

    python gerar_amostra.py               # só mostra o PLANO (buscas, período, cota). Não chama a API.
    python gerar_amostra.py --mock        # roda tudo com respostas simuladas (dados/mock/)
    python gerar_amostra.py --executar    # coleta REAL (gasta cota) e gera dados/amostra_rotulagem.csv
    python gerar_amostra.py --so-amostrar # refaz só o sorteio a partir de dados/brutos/

A amostra serve só para TESTE da perda de domínio (modelos treinados no RePro/Americanas).
Os dados ficam em dados/, que é ignorada pelo git (não redistribuir dados brutos).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from youtube import CUSTO, YouTube

# ============================== CONFIGURAÇÃO (fácil de trocar) ==============================
# consulta = texto enviado ao search.list; palavras = usadas para priorizar vídeos cujo título cita o produto
# "dias" (opcional) troca o período só daquele produto (útil para produtos com poucos vídeos por semana)
# Histórico: AirPods Pro 3 (vídeos quase todos em inglês) e Air fryer Mondial (poucos comentários)
# foram trocados em 09/11/2026 por Gelelé e iFood. O Gelelé também saiu: em 7 dias havia 3 vídeos e
# nenhum comentário válido; em 1 ano, os 31 vídeos que o citam somavam 7 comentários (a busca completava
# com vídeos genéricos de slime). Entrou Havaianas. Lição para o app: marca pequena = quase nada no YouTube.
PRODUTOS = [
    {"produto": "iPhone 17",          "categoria": "celular",          "consulta": "iPhone 17",
     "palavras": ["iphone 17", "iphone17"]},
    {"produto": "Havaianas",          "categoria": "calçado",          "consulta": "Havaianas",
     "palavras": ["havaianas"]},
    {"produto": "iFood",              "categoria": "app de entrega",   "consulta": "iFood",
     "palavras": ["ifood"]},
    # A busca por Malbec trouxe sobretudo vídeos de perfumes do Boticário em geral: tratado como a marca.
    {"produto": "O Boticário (perfumes)", "categoria": "perfume",      "consulta": "perfume Malbec Boticário",
     "palavras": ["malbec", "boticário", "boticario"]},
    {"produto": "PlayStation 5",      "categoria": "console",          "consulta": "PS5",
     "palavras": ["ps5", "playstation"]},
    {"produto": "BYD Dolphin Mini",   "categoria": "carro",            "consulta": "BYD Dolphin Mini",
     "palavras": ["dolphin", "byd"]},
]
DIAS = 7                      # período: os últimos DIAS completos (data de publicação do VÍDEO)
VIDEOS_BUSCADOS = 50          # 1 página do search.list (100 unidades, custa o mesmo para 5 ou 50)
MAX_VIDEOS_COM_COMENTARIOS = 25   # teto de vídeos lidos por produto (1 unidade cada)
COMENTARIOS_POR_VIDEO = 50    # 1 página do commentThreads.list
ALVO_BRUTO_POR_PRODUTO = 100  # para de ler vídeos ao juntar ~100 válidos aproveitáveis (≈600 no total)
APROVEITAVEIS_POR_VIDEO = 20  # cada vídeo conta no máximo 20 para o alvo (força variedade de vídeos)
AMOSTRA_POR_PRODUTO = 50      # ≈300 comentários na amostra
MAX_POR_VIDEO = 10
TITULOS_TOTAL = 50
MIN_PALAVRAS = 3
SEMENTE = 42
# ============================================================================================

PASTA = Path(__file__).resolve().parent
DADOS = PASTA / "dados"

# ------------------------------------------------------------------ filtros
_PT = set("""não nao é você vc voce que de do da dos das muito mto mais pra pro para com uma um eu isso
esse essa este esta tá ta está meu minha também tbm mas só bom boa ficou tem no na nos nas os ao aos
pq porque quando qual quem já ja agora ainda ser foi vai vou tô to sim nada tudo coisa melhor pior
cara caro barato comprei compra preço vale pena gostei adorei odiei acho achei kkk kkkk kkkkk mds
gente mano top demais aqui ele ela eles vocês voces dele dela depois antes sem sobre bem mal""".split())
_ES = set("""el los las y es muy pero por qué este también usted hola gracias mejor tiene con del lo
yo cuando bueno buena hermoso hermosa quiero ahora""".split())
_EN = set("""the and is this it i you of to for with that very good love my was are best ever
not but have so just like""".split())
_SPAM = re.compile(r"https?://|www\.|\.com\b|\.com\.br\b|meu canal|inscrev|whats ?app|telegram|"
                   r"renda extra|ganhe dinheiro|sorteio|segue de volta|sigo de volta", re.I)


def palavras(texto: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ]+", texto.lower())


def eh_portugues(texto: str) -> bool:
    """Detector leve por palavras funcionais (pt x es x en). Sem dependências."""
    t = texto.lower()
    if "ñ" in t or "¿" in t or "¡" in t:
        return False
    toks = palavras(t)
    pt = sum(w in _PT for w in toks) + 2 * bool(re.search(r"[ãõç]", t))
    es = sum(w in _ES for w in toks)
    en = sum(w in _EN for w in toks)
    return pt >= 1 and pt > es and pt > en


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return re.sub(r"\W+", " ", "".join(c for c in t if not unicodedata.combining(c))).strip()


def comentario_valido(texto: str) -> bool:
    return (len(palavras(texto)) >= MIN_PALAVRAS and not _SPAM.search(texto) and eh_portugues(texto))


# ------------------------------------------------------------------ período e plano
def periodo(dias: int, inicio: str | None = None, fim: str | None = None) -> tuple[str, str]:
    hoje = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    fim_dt = datetime.fromisoformat(fim).replace(tzinfo=timezone.utc) if fim else hoje
    ini_dt = datetime.fromisoformat(inicio).replace(tzinfo=timezone.utc) if inicio else fim_dt - timedelta(days=dias)
    f = "%Y-%m-%dT%H:%M:%SZ"
    return ini_dt.strftime(f), fim_dt.strftime(f)


def mostrar_plano(depois: str, antes: str) -> None:
    n = len(PRODUTOS)
    por_produto_max = CUSTO["search"] + -(-VIDEOS_BUSCADOS // 50) * CUSTO["videos"] + MAX_VIDEOS_COM_COMENTARIOS
    print("PLANO DA COLETA (nenhuma chamada foi feita)\n")
    print(f"Período (data de publicação do vídeo): {depois}  →  {antes}  ({DIAS} dias)")
    print("Parâmetros fixos do search.list: type=video, regionCode=BR, relevanceLanguage=pt, order=relevance, "
          f"maxResults={VIDEOS_BUSCADOS}\n")
    print(f"{'produto':22} {'categoria':17} consulta (q)")
    for p in PRODUTOS:
        extra = f"  (período próprio: desde {depois_do_produto(p, depois, antes)[:10]})" if "dias" in p else ""
        print(f"{p['produto']:22} {p['categoria']:17} \"{p['consulta']}\"{extra}")
    print(f"\nPor produto: 1 search.list ({CUSTO['search']} un.) + 1 videos.list (1 un., pula madeForKids) "
          f"+ até {MAX_VIDEOS_COM_COMENTARIOS} commentThreads.list (1 un. cada, {COMENTARIOS_POR_VIDEO} comentários, "
          f"textFormat=plainText, order=relevance),\n  parando ao juntar {ALVO_BRUTO_POR_PRODUTO} válidos (no máx. {APROVEITAVEIS_POR_VIDEO} contados por vídeo).")
    print(f"Estimativa: ~{n * (CUSTO['search'] + 1 + 10)} unidades (típico, ~10 vídeos lidos por produto) | máximo {n * por_produto_max} unidades "
          f"de 10.000/dia ({100 * n * por_produto_max / 10000:.1f}%).")
    print(f"Amostra: {AMOSTRA_POR_PRODUTO}/produto (≤{MAX_POR_VIDEO}/vídeo, pt, ≥{MIN_PALAVRAS} palavras, sem links/spam) "
          f"+ {TITULOS_TOTAL} títulos. Semente {SEMENTE}.")
    print("\nPara executar de verdade:  python gerar_amostra.py --executar")


# ------------------------------------------------------------------ coleta
def depois_do_produto(p: dict, depois: str, antes: str) -> str:
    """Início do período do produto: o geral, ou `antes - p['dias']` se o produto tiver período próprio."""
    if "dias" not in p:
        return depois
    fim = datetime.strptime(antes, "%Y-%m-%dT%H:%M:%SZ")
    return (fim - timedelta(days=p["dias"])).strftime("%Y-%m-%dT%H:%M:%SZ")


def coletar(yt: YouTube, depois: str, antes: str, produtos=PRODUTOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_todos, coments_todos = [], []
    for p in produtos:
        depois_p = depois_do_produto(p, depois, antes)
        print(f"\n== {p['produto']} ==" + (f"  (período próprio: {p['dias']} dias)" if "dias" in p else ""))
        achados = yt.buscar_videos(p["consulta"], depois_p, antes, max_videos=VIDEOS_BUSCADOS)
        detalhes = yt.detalhes_videos([v["video_id"] for v in achados])
        print(f"  {len(achados)} vídeos na busca, {len(detalhes)} após remover infantis")
        ordem = {v["video_id"]: i for i, v in enumerate(achados)}
        cita = lambda v: any(k in v["titulo"].lower() for k in p["palavras"])
        detalhes.sort(key=lambda v: (not cita(v), ordem.get(v["video_id"], 999)))

        validos, aproveitaveis, lidos = 0, 0, 0
        for v in detalhes:
            v.update(produto=p["produto"], categoria=p["categoria"], cita_produto=cita(v))
            videos_todos.append(v)
            if aproveitaveis >= ALVO_BRUTO_POR_PRODUTO or lidos >= MAX_VIDEOS_COM_COMENTARIOS or yt.cota_esgotada:
                continue
            if v["n_comentarios"] == 0:
                continue
            cs = yt.comentarios(v["video_id"], limite=COMENTARIOS_POR_VIDEO)
            lidos += 1
            n_ok = 0
            for c in cs:
                c.update(produto=p["produto"], valido=comentario_valido(c["texto"]))
                n_ok += c["valido"]
            validos += n_ok
            aproveitaveis += min(n_ok, APROVEITAVEIS_POR_VIDEO)
            coments_todos.extend(cs)
        print(f"  {lidos} vídeos lidos, {validos} comentários válidos ({aproveitaveis} aproveitáveis) | {yt.resumo()}")
        if yt.cota_esgotada:
            print("  !! cota esgotada: parando a coleta com o que já foi obtido")
            break
    return pd.DataFrame(videos_todos), pd.DataFrame(coments_todos)


# ------------------------------------------------------------------ amostragem
def _round_robin(grupos: dict[str, list], n: int, teto: int, rng: random.Random) -> list:
    """Pega itens alternando entre vídeos (≤ teto por vídeo) para espalhar a amostra."""
    filas = [list(g) for g in grupos.values()]
    rng.shuffle(filas)
    for f in filas:
        rng.shuffle(f)
    saida, rodada = [], 0
    while len(saida) < n and rodada < teto:
        for f in filas:
            if rodada < len(f) and len(saida) < n:
                saida.append(f[rodada])
        rodada += 1
    return saida


def amostrar(videos: pd.DataFrame, coments: pd.DataFrame, semente: int = SEMENTE) -> pd.DataFrame:
    rng = random.Random(semente)
    info = videos.drop_duplicates("video_id").set_index("video_id")
    c = coments[coments["valido"]].copy()
    c["_norm"] = c["texto"].map(normalizar)
    c = c.drop_duplicates("_norm")

    linhas = []
    for produto in [p["produto"] for p in PRODUTOS]:
        sub = c[c["produto"] == produto]
        grupos = defaultdict(list)
        for r in sub.to_dict("records"):
            grupos[r["video_id"]].append(r)
        escolhidos = _round_robin(grupos, AMOSTRA_POR_PRODUTO, MAX_POR_VIDEO, rng)
        if len(escolhidos) < AMOSTRA_POR_PRODUTO:
            print(f"  aviso: {produto} tem só {len(escolhidos)} comentários elegíveis")
        for r in escolhidos:
            linhas.append({"tipo": "comentário", "produto": produto, "video_id": r["video_id"], "comment_id": r["comment_id"],
                           "texto": r["texto"], "likes": r["likes"]})

    # títulos: ~TITULOS_TOTAL divididos entre os produtos, só em pt e sem repetição
    v = videos.drop_duplicates("video_id")
    v = v[v["titulo"].map(eh_portugues)]
    v = v.assign(_norm=v["titulo"].map(normalizar)).drop_duplicates("_norm")
    cotas = [TITULOS_TOTAL // len(PRODUTOS) + (i < TITULOS_TOTAL % len(PRODUTOS)) for i in range(len(PRODUTOS))]
    for p, cota in zip(PRODUTOS, cotas):
        cand = v[v["produto"] == p["produto"]].to_dict("records")
        rng.shuffle(cand)
        cand.sort(key=lambda r: not r["cita_produto"])  # prioriza títulos que citam o produto
        for r in cand[:cota]:
            linhas.append({"tipo": "título", "produto": p["produto"], "video_id": r["video_id"], "comment_id": "",
                           "texto": r["titulo"], "likes": r["likes"]})

    df = pd.DataFrame(linhas)
    df["titulo_video"] = df["video_id"].map(info["titulo"])
    df["data_video"] = df["video_id"].map(info["data_video"]).str[:10]
    df = df.sample(frac=1, random_state=semente).reset_index(drop=True)  # embaralha produtos e tipos
    cont = {"comentário": 0, "título": 0}
    ids = []
    for t in df["tipo"]:
        cont[t] += 1
        ids.append(f"{'c' if t == 'comentário' else 't'}{cont[t]:03d}")
    df.insert(0, "id", ids)
    return df[["id", "tipo", "produto", "video_id", "comment_id", "titulo_video", "data_video", "texto", "likes"]]


def salvar(saida: Path, videos: pd.DataFrame | None, coments: pd.DataFrame | None, amostra: pd.DataFrame) -> None:
    (saida / "brutos").mkdir(parents=True, exist_ok=True)
    if videos is not None:
        videos.to_csv(saida / "brutos" / "videos.csv", index=False, encoding="utf-8-sig")
        coments.to_csv(saida / "brutos" / "comentarios.csv", index=False, encoding="utf-8-sig")
    amostra.to_csv(saida / "amostra_rotulagem.csv", index=False, encoding="utf-8-sig")
    print(f"\nAmostra: {len(amostra)} itens -> {saida / 'amostra_rotulagem.csv'}")
    print(amostra.groupby(["produto", "tipo"]).size().unstack(fill_value=0).to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--executar", action="store_true", help="coleta real (gasta cota)")
    modo.add_argument("--mock", action="store_true", help="respostas simuladas, saída em dados/mock/")
    modo.add_argument("--so-amostrar", action="store_true", help="refaz o sorteio a partir de dados/brutos/")
    ap.add_argument("--inicio", help="AAAA-MM-DD (padrão: hoje - DIAS)")
    ap.add_argument("--fim", help="AAAA-MM-DD, exclusivo (padrão: hoje 00:00 UTC)")
    a = ap.parse_args(argv)
    depois, antes = periodo(DIAS, a.inicio, a.fim)

    if a.so_amostrar:
        videos = pd.read_csv(DADOS / "brutos" / "videos.csv", encoding="utf-8-sig")
        coments = pd.read_csv(DADOS / "brutos" / "comentarios.csv", encoding="utf-8-sig")
        salvar(DADOS, None, None, amostrar(videos, coments))
        return
    if not (a.executar or a.mock):
        mostrar_plano(depois, antes)
        return

    if a.mock:
        from mock_api import criar_transporte
        saida = DADOS / "mock"
        yt = YouTube(transporte=criar_transporte(), pasta_cache=None)
    else:
        saida = DADOS
        yt = YouTube(verbose=True)
    videos, coments = coletar(yt, depois, antes)
    if coments.empty:
        sys.exit("Nenhum comentário coletado. " + yt.resumo())
    amostra = amostrar(videos, coments)
    salvar(saida, videos, coments, amostra)
    log = {"quando": datetime.now().isoformat(timespec="seconds"), "periodo": [depois, antes],
           "produtos": [p["consulta"] for p in PRODUTOS], "unidades": yt.unidades,
           "chamadas_reais": yt.chamadas_reais, "cache": yt.acertos_cache, "cota_esgotada": yt.cota_esgotada,
           "videos": len(videos), "comentarios_brutos": len(coments),
           "comentarios_validos": int(coments["valido"].sum()), "itens_amostra": len(amostra)}
    (saida / "coleta_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + yt.resumo())


if __name__ == "__main__":
    main()
