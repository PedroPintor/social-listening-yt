"""Radar de Marca: o que o público diz do seu produto no YouTube (página única).

    python -m streamlit run src/app.py

Precisa do classificador treinado (python src/treinar_classificador.py) e, para análises
novas, da chave YOUTUBE_API_KEY no .env da raiz. Análises salvas abrem sem chave.
"""
from datetime import date, timedelta
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

import pipeline

NOME = "Radar de Marca"
st.set_page_config(page_title=NOME, page_icon="📡", layout="wide")

CORES = {"positivo": "#2E9E5B", "neutro": "#9AA0A6", "negativo": "#D64545"}
ORDEM = ["positivo", "neutro", "negativo"]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {border-radius: 20px; padding: 48px 44px; color: #fff; margin-bottom: 8px;
       background: radial-gradient(circle at 85% 20%, #ff5c5c 0%, transparent 45%),
                   linear-gradient(120deg, #1d2b64 0%, #3b2a8f 55%, #8e2de2 100%);}
.hero .selo {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem;
             background: rgba(255,255,255,0.16); letter-spacing: .04em;}
.hero h1 {font-size: 3rem; margin: 14px 0 6px; color: #fff; line-height: 1.1;}
.hero p {font-size: 1.2rem; max-width: 640px; opacity: .92; margin: 0;}
.passos {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0 8px;}
.passo {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 18px;
        background: rgba(128,128,128,.06);}
.passo .ic {font-size: 1.6rem;} .passo b {display: block; margin: 6px 0 4px;}
.passo span {font-size: .9rem; opacity: .8;}
.etapas {list-style: none; padding: 0; margin: 6px 0 0;}
.etapas li {padding: 7px 0; font-size: 1rem; display: flex; align-items: center; gap: 10px;}
.etapas .pend {opacity: .45;} .etapas small {opacity: .7;}
.gira {width: 16px; height: 16px; border-radius: 50%; border: 3px solid rgba(128,128,128,.3);
       border-top-color: #8e2de2; animation: gira 0.8s linear infinite; display: inline-block;}
@keyframes gira {to {transform: rotate(360deg);}}
.barra-rot {display: flex; justify-content: space-between; font-size: .95rem; margin: 14px 0 6px;}
.barra {display: flex; height: 34px; border-radius: 10px; overflow: hidden;}
.barra div {display: flex; align-items: center; justify-content: center; color: #fff;
            font-size: .85rem; font-weight: 600; white-space: nowrap;}
.legenda {display: flex; gap: 18px; font-size: .85rem; margin-top: 10px; opacity: .85;}
.legenda i {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px;}
.videos {display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 16px;}
.video {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; overflow: hidden;
        background: rgba(128,128,128,.05); border-top: 5px solid var(--cor);}
.video img {width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block;}
.video .corpo {padding: 12px 14px 14px;}
.video a {color: inherit; text-decoration: none;}
.video .tit {font-weight: 600; font-size: .95rem; line-height: 1.3; height: 2.6em; overflow: hidden;}
.video .meta {font-size: .8rem; opacity: .7; margin: 6px 0 10px;}
.video .selo {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem;
              font-weight: 600; color: #fff; background: var(--cor); margin-bottom: 8px;}
.video .mini {display: flex; height: 8px; border-radius: 4px; overflow: hidden;}
.coments {display: flex; flex-direction: column; gap: 10px;}
.coment {border: 1px solid rgba(128,128,128,.22); border-left: 5px solid var(--cor); border-radius: 10px;
         padding: 12px 16px; background: rgba(128,128,128,.05);}
.coment .topo {display: flex; justify-content: space-between; font-size: .8rem; opacity: .85; margin-bottom: 6px;}
.coment .sent {color: var(--cor); font-weight: 700; text-transform: uppercase; letter-spacing: .04em;}
.coment .txt {font-size: 1rem; line-height: 1.5;}
.coment .vid {font-size: .78rem; opacity: .6; margin-top: 6px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Carregando o classificador de sentimento…")
def carregar_classificador():
    from sentimento import ClassificadorSentimento
    return ClassificadorSentimento()


@st.cache_resource(show_spinner="Carregando o modelo de NER…")
def carregar_ner():
    from gliner2 import AutoExtractor
    return AutoExtractor.from_pretrained("ottema/gliner2-ptbr")


def compacto(html: str) -> str:
    """HTML em uma linha: o Markdown do Streamlit trata linhas recuadas como bloco de código."""
    return " ".join(linha.strip() for linha in html.splitlines() if linha.strip())


def barra_html(contagem: dict, altura_mini=False) -> str:
    total = sum(contagem.values()) or 1
    partes = []
    for s in ORDEM:
        p = contagem.get(s, 0) / total * 100
        if p > 0:
            txt = "" if altura_mini or p < 7 else f"{p:.0f}%"
            partes.append(f'<div style="width:{p}%;background:{CORES[s]}" title="{s}: {p:.0f}%">{txt}</div>')
    return f'<div class="{"mini" if altura_mini else "barra"}">{"".join(partes)}</div>'


def clima(pos: float, neg: float, n: int):
    """Selo do vídeo pelo saldo de positivos e negativos dos comentários."""
    if n < 5:
        return "Poucos comentários", "#9AA0A6"
    if pos - neg >= 0.10:
        return "😀 Positivo", CORES["positivo"]
    if neg - pos >= 0.10:
        return "😠 Negativo", CORES["negativo"]
    return "😐 Dividido", "#C9A227"


# ------------------------------------------------------------------ abertura
st.markdown(compacto(f"""
<div class="hero">
  <span class="selo">📡 Monitoramento de marca · YouTube Brasil</span>
  <h1>{NOME}</h1>
  <p>Descubra em minutos o que o público está dizendo do seu produto nos comentários dos vídeos mais recentes.</p>
</div>
<div class="passos">
  <div class="passo"><div class="ic">🔎</div><b>Coleta</b><span>Os vídeos publicados no período e os comentários deles, direto da API do YouTube.</span></div>
  <div class="passo"><div class="ic">🎯</div><b>Separação</b><span>Destaca os comentários que citam o seu produto pelo nome.</span></div>
  <div class="passo"><div class="ic">🧠</div><b>Sentimento</b><span>Uma IA treinada em português classifica cada comentário: positivo, neutro ou negativo.</span></div>
  <div class="passo"><div class="ic">🏷️</div><b>Concorrência</b><span>Reconhecimento de entidades mostra quais outras marcas aparecem na conversa.</span></div>
</div>
"""), unsafe_allow_html=True)

# ------------------------------------------------------------------ consulta
st.subheader("Analise um produto")
with st.form("consulta"):
    c1, c2, c3 = st.columns([2, 3, 2])
    produto = c1.text_input("Produto", placeholder="ex.: iPhone 17")
    apelidos = c2.text_input("Apelidos (separados por vírgula)", placeholder="ex.: iphone, apple, ip17",
                             help="Usados para destacar os comentários que citam o produto.")
    hoje = date.today()
    periodo = c3.date_input("Vídeos publicados entre", (hoje - timedelta(days=2), hoje - timedelta(days=1)),
                            max_value=hoje, format="DD/MM/YYYY")
    with st.expander("Opções da coleta"):
        o1, o2 = st.columns(2)
        max_videos = o1.slider("Máximo de vídeos", 1, 20, 5)
        max_coment = o2.slider("Comentários por vídeo", 20, 200, 60, step=20)
        st.caption(f"Custo máximo na cota da API: ~{pipeline.custo_estimado(max_videos, max_coment)} "
                   "de 10.000 unidades por dia. Repetir a mesma consulta usa o cache e não gasta cota.")
    enviar = st.form_submit_button("Analisar", type="primary", use_container_width=True)

salvas = pipeline.analises_salvas()
if salvas:
    escolha = st.selectbox("…ou abra uma análise salva", ["—"] + [p.stem for p in salvas])
    if escolha != "—" and st.session_state.get("aberta") != escolha:
        st.session_state["resultado"] = pipeline.abrir(pipeline.PASTA_ANALISES / f"{escolha}.json")
        st.session_state["aberta"] = escolha


def painel_etapas(lugar, estado):
    itens = []
    for chave, texto in pipeline.ETAPAS.items():
        situacao, frac, detalhe = estado.get(chave, ("pendente", 0.0, ""))
        if situacao == "feito":
            itens.append(f"<li>✅ {texto} <small>· {escape(detalhe)}</small></li>")
        elif situacao == "rodando":
            pct = f" <small>· {frac:.0%}</small>" if 0 < frac < 1 else ""
            itens.append(f'<li><span class="gira"></span> <b>{texto}</b>{pct}</li>')
        else:
            itens.append(f'<li class="pend">○ {texto}</li>')
    lugar.markdown(f'<ul class="etapas">{"".join(itens)}</ul>', unsafe_allow_html=True)


if enviar:
    if not produto.strip():
        st.warning("Digite o nome do produto.")
    elif not (isinstance(periodo, tuple) and len(periodo) == 2):
        st.warning("Escolha a data inicial e a final do período.")
    else:
        try:
            with st.container(border=True):
                st.markdown(f"**Analisando {escape(produto.strip())}…**")
                lugar = st.empty()
                estado = {}
                painel_etapas(lugar, estado)
                clf, ner = carregar_classificador(), carregar_ner()

                def avisar(etapa, frac, detalhe=""):
                    chaves = list(pipeline.ETAPAS)
                    for anterior in chaves[:chaves.index(etapa)]:       # etapas anteriores já terminaram
                        if estado.get(anterior, ("",))[0] != "feito":
                            estado[anterior] = ("feito", 1.0, estado.get(anterior, ("", 0, ""))[2])
                    estado[etapa] = ("feito", 1.0, detalhe) if frac >= 1 else ("rodando", frac, detalhe)
                    painel_etapas(lugar, estado)

                r = pipeline.analisar(produto.strip(), apelidos, periodo[0], periodo[1], max_videos, max_coment,
                                      clf, ner, avisar=avisar)
            pipeline.salvar(r)
            st.session_state["resultado"], st.session_state["aberta"] = r, None
        except RuntimeError as e:          # chave ausente no .env
            st.error(f"{e} Para testar sem chave, abra uma análise salva.")
        except FileNotFoundError as e:     # classificador não treinado
            st.error(str(e))

r = st.session_state.get("resultado")
if r is None:
    st.stop()

# ------------------------------------------------------------------ resultado
V, C = r["videos"], r["comentarios"]
fmt = lambda d: pd.to_datetime(d).strftime("%d/%m/%Y")
st.divider()
st.header(f"{r['produto']} no YouTube")
st.caption(f"Vídeos publicados de {fmt(r['inicio'])} a {fmt(r['fim'])} · busca: {', '.join(r['termos'])} · "
           f"análise de {pd.to_datetime(r['gerado_em']).strftime('%d/%m/%Y %H:%M')}")
if r.get("cota_esgotada"):
    st.warning("A cota diária da API acabou no meio da coleta: o resultado está incompleto.")
if C.empty:
    st.info("Nenhum comentário encontrado. Tente um período maior ou outro nome para o produto.")
    st.stop()

cita = C[C["cita_produto"]]
pct = lambda df, s: (df["sentimento"] == s).mean() * 100 if len(df) else 0
m = st.columns(5)
m[0].metric("Vídeos", len(V))
m[1].metric("Comentários lidos", len(C))
m[2].metric("Citam o produto", f"{len(cita)} ({len(cita) / len(C):.0%})")
m[3].metric("Positivos (citam)", f"{pct(cita, 'positivo'):.0f}%")
m[4].metric("Negativos (citam)", f"{pct(cita, 'negativo'):.0f}%")

# sentimento por grupo
st.subheader("Sentimento dos comentários")
html = []
for nome, df in [("Citam o produto", cita), ("Demais comentários", C[~C["cita_produto"]])]:
    html.append(f'<div class="barra-rot"><b>{nome}</b><span>{len(df)} comentários</span></div>')
    html.append(barra_html(df["sentimento"].value_counts().to_dict()) if len(df) else
                '<div class="barra" style="background:rgba(128,128,128,.15)"></div>')
leg = "".join(f'<span><i style="background:{CORES[s]}"></i>{s}</span>' for s in ORDEM)
st.markdown("".join(html) + f'<div class="legenda">{leg}</div>', unsafe_allow_html=True)
st.caption("**Citam o produto:** quase todos são opinião sobre o produto. **Demais comentários:** muitos falam do "
           "vídeo ou do youtuber, então o sentimento deles não é necessariamente sobre o produto.")

# vídeos
st.subheader("Vídeos")
cards = []
for _, v in V.iterrows():
    cv = C[C["video_id"] == v["video_id"]]
    cont = cv["sentimento"].value_counts().to_dict()
    pos, neg = (cv["sentimento"] == "positivo").mean() if len(cv) else 0, (cv["sentimento"] == "negativo").mean() if len(cv) else 0
    selo, cor = clima(pos, neg, len(cv))
    link = f"https://www.youtube.com/watch?v={v['video_id']}"
    cards.append(f"""
<div class="video" style="--cor:{cor}"><a href="{link}" target="_blank">
  <img src="https://i.ytimg.com/vi/{v['video_id']}/mqdefault.jpg" alt="">
  <div class="corpo"><span class="selo">{selo}</span>
    <div class="tit">{escape(str(v['titulo']))}</div>
    <div class="meta">{escape(str(v['canal']))} · {pd.to_datetime(v['data_video']).strftime('%d/%m')} ·
      {f"{int(v['views']):,}".replace(",", ".")} views · {len(cv)} comentários lidos</div>
    {barra_html(cont, altura_mini=True)}
  </div></a></div>""")
st.markdown(compacto(f'<div class="videos">{"".join(cards)}</div>'), unsafe_allow_html=True)
st.caption("O selo resume os comentários de cada vídeo: positivo ou negativo quando um lado supera o outro em "
           "10 pontos percentuais.")

# comentários
st.subheader("Comentários")
f1, f2, f3 = st.columns([3, 3, 2])
grupo_sel = f1.radio("Mostrar", ["Citam o produto", "Todos", "Demais"], horizontal=True)
sent_sel = f2.multiselect("Sentimento", ORDEM, default=ORDEM)
ordem_sel = f3.selectbox("Ordenar por", ["Mais curtidos", "Mais recentes"])
vis = C if grupo_sel == "Todos" else C[C["cita_produto"] == (grupo_sel == "Citam o produto")]
vis = vis[vis["sentimento"].isin(sent_sel)].merge(V[["video_id", "titulo"]], on="video_id", how="left")
vis = vis.sort_values("likes" if ordem_sel == "Mais curtidos" else "data_comentario", ascending=False)
st.caption(f"{len(vis)} comentários")
itens = []
for _, c in vis.iterrows():
    s = c["sentimento"]
    data = pd.to_datetime(c["data_comentario"]).strftime("%d/%m") if c.get("data_comentario") else ""
    itens.append(
        f'<div class="coment" style="--cor:{CORES[s]}">'
        f'<div class="topo"><span class="sent">{s}</span><span>👍 {int(c["likes"])} · {data}</span></div>'
        f'<div class="txt">{escape(str(c["texto"])).replace(chr(10), "<br>")}</div>'
        f'<div class="vid">em: {escape(str(c["titulo"]))}</div></div>')
with st.container(height=560, border=False):
    st.markdown(f'<div class="coments">{"".join(itens)}</div>', unsafe_allow_html=True)

# marcas citadas
st.subheader("Marcas citadas")
ents = pipeline.contar_entidades(C, r["termos"])
if ents.empty:
    st.caption("Nenhuma marca encontrada nos comentários.")
else:
    ents["tipo"] = ents["é o produto pesquisado"].map({True: "Produto pesquisado", False: "Outras marcas"})
    st.altair_chart(alt.Chart(ents).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("comentarios:Q", title="comentários em que aparece"),
        y=alt.Y("entidade:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
        color=alt.Color("tipo:N", scale=alt.Scale(domain=["Produto pesquisado", "Outras marcas"],
                        range=["#3b2a8f", "#C9A227"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["entidade", "comentarios"]).properties(height=alt.Step(26)),
        use_container_width=True)
    st.caption("Marcas e produtos reconhecidos nos comentários. Em amarelo, outras marcas: concorrentes, acessórios, lojas.")
