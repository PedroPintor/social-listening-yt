"""Concordância entre os dois rotuladores (kappa de Cohen) e exportação das divergências.

    python kappa.py dados/rotulos_pedro.csv dados/rotulos_maria.csv
    python kappa.py A.csv B.csv --amostra dados/amostra_rotulagem.csv --saida dados/divergencias.csv

Os CSVs de rótulos vêm do rotulador.html (colunas id, rotulo).

Saídas:
- dados/divergencias.csv (com texto, NÃO versionar): preencha a coluna rotulo_final na reunião de consenso.
  Rodar de novo preserva o que já foi preenchido.
- rotulos/rotulos_consolidados.csv (SEM texto, versionado no git): id, tipo, produto, video_id, comment_id,
  rótulo de cada pessoa e rotulo_final. Os textos podem ser recuperados pela API a partir dos IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROTULOS = ["positivo", "neutro", "negativo", "misto", "nao_sobre_produto"]
PASTA = Path(__file__).resolve().parent


def ler(caminho: str) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(caminho, encoding="utf-8-sig", dtype=str)
    nome = Path(caminho).stem.removeprefix("rotulos_")
    return nome, df[["id", "rotulo"]].dropna()


def interpretar(k: float) -> str:  # Landis & Koch (1977)
    for limite, txt in [(0.2, "leve"), (0.4, "razoável"), (0.6, "moderada"), (0.8, "substancial")]:
        if k <= limite:
            return txt
    return "quase perfeita"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rotulos_a")
    ap.add_argument("rotulos_b")
    ap.add_argument("--amostra", default=str(PASTA / "dados" / "amostra_rotulagem.csv"))
    ap.add_argument("--saida", default=str(PASTA / "dados" / "divergencias.csv"))
    ap.add_argument("--consolidado", default=str(PASTA / "rotulos" / "rotulos_consolidados.csv"))
    a = ap.parse_args(argv)

    na, A = ler(a.rotulos_a)
    nb, B = ler(a.rotulos_b)
    if na == nb:
        na, nb = na + "_A", nb + "_B"
    m = A.merge(B, on="id", suffixes=(f"_{na}", f"_{nb}"))
    ca, cb = f"rotulo_{na}", f"rotulo_{nb}"
    print(f"{na}: {len(A)} rótulos | {nb}: {len(B)} rótulos | em comum: {len(m)}")
    if m.empty:
        raise SystemExit("Nenhum id em comum entre os dois arquivos.")

    amostra = None
    if Path(a.amostra).exists():
        amostra = pd.read_csv(a.amostra, encoding="utf-8-sig", dtype=str)
        cols = [c for c in ["id", "tipo", "produto", "video_id", "comment_id", "titulo_video", "texto"] if c in amostra]
        m = m.merge(amostra[cols], on="id", how="left")

    k = cohen_kappa_score(m[ca], m[cb], labels=ROTULOS)
    print(f"\nConcordância bruta: {(m[ca] == m[cb]).mean():.1%}")
    print(f"Kappa de Cohen (5 classes): {k:.3f}  ({interpretar(k)})")

    # visão só de polaridade: itens que ambos consideraram sobre o produto e não mistos
    pol = m[m[ca].isin(ROTULOS[:3]) & m[cb].isin(ROTULOS[:3])]
    if len(pol) > 1:
        kp = cohen_kappa_score(pol[ca], pol[cb], labels=ROTULOS[:3])
        print(f"Kappa só pos/neu/neg ({len(pol)} itens): {kp:.3f}  ({interpretar(kp)})")
    if "tipo" in m:
        for tipo, g in m.groupby("tipo"):
            if len(g) > 1:
                print(f"Kappa em '{tipo}' ({len(g)} itens): {cohen_kappa_score(g[ca], g[cb], labels=ROTULOS):.3f}")

    matriz = pd.crosstab(pd.Categorical(m[ca], ROTULOS), pd.Categorical(m[cb], ROTULOS),
                         rownames=[na], colnames=[nb], dropna=False)
    print(f"\nMatriz de concordância (linhas = {na}, colunas = {nb}):")
    print(matriz.to_string())

    # divergências: preserva rotulo_final já preenchido numa execução anterior
    resolvidos = {}
    if Path(a.saida).exists():
        antigo = pd.read_csv(a.saida, encoding="utf-8-sig", dtype=str).fillna("")
        if "rotulo_final" in antigo:
            resolvidos = {i: r.strip() for i, r in zip(antigo["id"], antigo["rotulo_final"]) if r.strip()}
    invalidos = sorted(set(resolvidos.values()) - set(ROTULOS))
    if invalidos:
        raise SystemExit(f"rotulo_final com valor inválido em {a.saida}: {invalidos}. Use um de {ROTULOS}.")
    div = m[m[ca] != m[cb]].copy()
    div["rotulo_final"] = div["id"].map(lambda i: resolvidos.get(i, ""))
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    div.to_csv(a.saida, index=False, encoding="utf-8-sig")
    falta = int((div["rotulo_final"] == "").sum())
    print(f"\n{len(div)} divergências em {a.saida} ({len(div) - falta} resolvidas, "
          f"{falta} a resolver na coluna rotulo_final).")

    # tabela versionada, SEM texto
    cons = m.copy()
    cons["rotulo_final"] = cons[ca].where(cons[ca] == cons[cb], cons["id"].map(lambda i: resolvidos.get(i, "")))
    sem_texto = [c for c in ["id", "tipo", "produto", "video_id", "comment_id"] if c in cons] + [ca, cb, "rotulo_final"]
    cons = cons[sem_texto].sort_values("id")
    Path(a.consolidado).parent.mkdir(parents=True, exist_ok=True)
    cons.to_csv(a.consolidado, index=False, encoding="utf-8")
    print(f"Tabela sem texto (pode ir para o git): {a.consolidado}")


if __name__ == "__main__":
    main()
