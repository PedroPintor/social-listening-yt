"""Fluxo do app: coleta → separação pelo nome → sentimento → marcas citadas (NER).

Decisões e evidências em `docs/notas/decisoes_de_escopo.md` e no notebook
`src-pln/notebooks/05c_fluxo_do_app.ipynb`:
- separação dos comentários pela busca do nome do produto e apelidos (precisão 0,91);
  nenhum comentário é descartado;
- sentimento com o BERTimbau congelado + Regressão Logística;
- NER (GLiNER2) só para listar as marcas e produtos citados, porque como filtro de
  relevância ele não separou o ruído.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
# BERTimbau e GLiNER2 são baixados do Hugging Face para esta pasta do projeto (fora do git).
# Precisa vir antes de importar o transformers ou o gliner2; apagar a pasta remove os modelos.
os.environ.setdefault("HF_HOME", str(RAIZ / "modelos_baixados"))
sys.path.insert(0, str(RAIZ / "src-pln" / "coleta_youtube"))
from youtube import YouTube, estimar_unidades  # noqa: E402

PASTA_ANALISES = RAIZ / "src" / "analises"          # análises salvas (fora do git: têm textos do YouTube)
CATEGORIAS_NER = ["marca", "produto", "empresa"]
LIMIAR_NER = 0.5
FUSO_BR = "-03:00"                                   # o período é o dia no horário de Brasília


# ---------------------------------------------------------------- separação pelo nome
def sem_acento(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", str(s).lower()) if unicodedata.category(ch) != "Mn")


def lista_apelidos(produto: str, apelidos: str) -> list[str]:
    """'iPhone 17' + 'iphone, apple' -> ['iphone 17', 'iphone', 'apple'] (sem acento, sem repetição)."""
    termos = [produto] + [a for a in re.split(r"[,;\n]", apelidos or "")]
    vistos = []
    for t in (sem_acento(t).strip() for t in termos):
        if t and t not in vistos:
            vistos.append(t)
    return vistos


def cita_produto(texto: str, termos: list[str]) -> bool:
    t = sem_acento(texto)
    return any(re.search(rf"(?<!\w){re.escape(a)}s?(?!\w)", t) for a in termos)   # aceita plural simples


# ---------------------------------------------------------------- coleta
def custo_estimado(max_videos: int, max_comentarios: int) -> int:
    return estimar_unidades(1, 1, max_videos, -(-max_comentarios // 100))


ETAPAS = {                       # chave -> texto mostrado na tela, na ordem do fluxo
    "busca": "Buscando vídeos publicados no período",
    "comentarios": "Lendo os comentários de cada vídeo",
    "separacao": "Separando quem cita o produto",
    "sentimento": "Classificando o sentimento com IA (BERTimbau)",
    "ner": "Encontrando marcas e concorrentes citados (NER)",
}


def coletar(produto: str, inicio: date, fim: date, max_videos: int, max_comentarios: int,
            avisar=lambda etapa, frac, detalhe="": None):
    """Vídeos publicados entre `inicio` e `fim` (inclusive, horário de Brasília) e os comentários deles."""
    yt = YouTube()
    depois = f"{inicio.isoformat()}T00:00:00{FUSO_BR}"
    antes = f"{(fim + timedelta(days=1)).isoformat()}T00:00:00{FUSO_BR}"
    avisar("busca", 0.0)
    achados = yt.buscar_videos(produto, depois, antes, max_videos=min(50, max_videos * 2))
    videos = yt.detalhes_videos([v["video_id"] for v in achados])       # sem os vídeos infantis
    # a região BR não garante vídeos brasileiros: descarta os com áudio declarado em outra língua
    # (os que não declaram o idioma ficam) e os sem comentários
    videos = [v for v in videos if v["n_comentarios"]
              and (not v["idioma_audio"] or v["idioma_audio"].lower().startswith("pt"))][:max_videos]
    avisar("busca", 1.0, f"{len(videos)} vídeos")
    comentarios = []
    for i, v in enumerate(videos):
        avisar("comentarios", i / max(len(videos), 1), f"vídeo {i + 1} de {len(videos)}")
        comentarios += yt.comentarios(v["video_id"], limite=max_comentarios)
    avisar("comentarios", 1.0, f"{len(comentarios)} comentários")
    return pd.DataFrame(videos), pd.DataFrame(comentarios), yt


# ---------------------------------------------------------------- NER
def normalizar_entidade(e: str) -> str:
    return re.sub(r"^\W+|\W+$", "", str(e).strip()).lower()


def extrair_entidades(textos, extrator, avisar=lambda frac: None) -> list[list[str]]:
    """Para cada texto, a lista de marcas/produtos/empresas encontradas (normalizadas, sem repetição)."""
    saida = []
    for i, texto in enumerate(textos):
        ents = extrator.extract_entities(str(texto), CATEGORIAS_NER, threshold=LIMIAR_NER)["entities"]
        nomes = {normalizar_entidade(e) for lista in ents.values() for e in lista}
        saida.append(sorted(n for n in nomes if len(n) > 1))
        avisar((i + 1) / len(textos))
    if not saida:
        avisar(1.0)
    return saida


def contar_entidades(comentarios: pd.DataFrame, termos: list[str], top: int = 15) -> pd.DataFrame:
    """Marcas citadas: em quantos comentários aparece cada entidade (o próprio produto é marcado)."""
    cont = comentarios["entidades"].explode().dropna().value_counts().head(top)
    df = cont.rename_axis("entidade").reset_index(name="comentarios")
    df["é o produto pesquisado"] = [cita_produto(e, termos) for e in df["entidade"]]
    return df


# ---------------------------------------------------------------- análise completa
def analisar(produto, apelidos, inicio, fim, max_videos, max_comentarios, classificador, extrator,
             avisar=lambda etapa, frac, detalhe="": None) -> dict:
    """`avisar(etapa, fração, detalhe)` é chamado a cada passo; `etapa` é uma chave de ETAPAS."""
    termos = lista_apelidos(produto, apelidos)
    videos, comentarios, yt = coletar(produto, inicio, fim, max_videos, max_comentarios, avisar=avisar)
    if comentarios.empty:
        comentarios = pd.DataFrame(columns=["video_id", "comment_id", "texto", "likes", "data_comentario"])
    comentarios["cita_produto"] = [cita_produto(t, termos) for t in comentarios["texto"]]
    avisar("separacao", 1.0, f"{int(comentarios['cita_produto'].sum())} citam o produto")
    avisar("sentimento", 0.0)
    comentarios["sentimento"] = classificador.prever(
        comentarios["texto"].tolist(), progresso=lambda f: avisar("sentimento", f))
    avisar("sentimento", 1.0, f"{len(comentarios)} comentários classificados")
    avisar("ner", 0.0)
    comentarios["entidades"] = extrair_entidades(
        comentarios["texto"].tolist(), extrator, avisar=lambda f: avisar("ner", f))
    avisar("ner", 1.0, f"{comentarios['entidades'].explode().nunique()} marcas diferentes")
    return {
        "produto": produto, "termos": termos, "inicio": inicio.isoformat(), "fim": fim.isoformat(),
        "gerado_em": datetime.now().isoformat(timespec="minutes"),
        "unidades_api": yt.unidades, "cota_esgotada": yt.cota_esgotada,
        "videos": videos, "comentarios": comentarios,
    }


# ---------------------------------------------------------------- salvar / abrir
def _nome_arquivo(r: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", sem_acento(r["produto"])).strip("-")
    return f"{slug}_{r['inicio']}_{r['fim']}.json"


def salvar(r: dict, pasta: Path = PASTA_ANALISES) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    dados = {k: v for k, v in r.items() if k not in ("videos", "comentarios")}
    dados["videos"] = r["videos"].to_dict(orient="records")
    dados["comentarios"] = r["comentarios"].to_dict(orient="records")
    arq = pasta / _nome_arquivo(r)
    arq.write_text(json.dumps(dados, ensure_ascii=False, default=str), encoding="utf-8")
    return arq


def abrir(arq: Path) -> dict:
    r = json.loads(Path(arq).read_text(encoding="utf-8"))
    r["videos"] = pd.DataFrame(r["videos"])
    r["comentarios"] = pd.DataFrame(r["comentarios"])
    return r


def analises_salvas(pasta: Path = PASTA_ANALISES) -> list[Path]:
    return sorted(pasta.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if pasta.exists() else []
