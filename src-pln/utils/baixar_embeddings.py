"""Baixa e filtra os embeddings pré-treinados usados na etapa 4 (rede neural).

Embeddings:
  glove_nilc     NILC-USP GloVe 300d (cópia oficial do NILC no Hugging Face,
                 nilc-nlp/glove-300d; o site nilc.icmc.usp.br estava fora do ar)
                 ~1,12 GB (safetensors) + 9 MB (vocabulário). Licença CC BY 4.0.
  fasttext_fb    FastText do Facebook para português, cc.pt.300.vec.gz
                 (Common Crawl + Wikipedia), ~1,27 GB. Licença CC BY-SA 3.0.

Os arquivos brutos são grandes, então o script guarda só um recorte em
data/embeddings/<nome>_filtrado.npz com:
  - todas as palavras que aparecem no RePro (e a forma sem acento de cada uma,
    para comparar normalizações no notebook);
  - as N palavras mais frequentes do embedding (padrão 50.000), para a rede
    ter vetores de palavras que não aparecem no RePro (gírias, por exemplo).
O npz tem `palavras` (str), `vetores` (float32) e `rank` (posição da palavra
no arquivo original; menor = mais frequente). Por padrão os brutos são apagados
depois da filtragem.

Uso (a partir de src-pln/):
    python utils/baixar_embeddings.py                    # baixa e filtra os dois
    python utils/baixar_embeddings.py glove_nilc         # só um
    python utils/baixar_embeddings.py --manter-brutos    # não apaga os originais
    python utils/baixar_embeddings.py --forcar           # refaz mesmo se já existir
"""
import argparse
import gzip
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.dados import carregar_repro, tokenizar  # noqa: E402

PASTA_EMB = RAIZ / "data" / "embeddings"
PASTA_BRUTOS = PASTA_EMB / "brutos"
TOP_N = 50_000

HF = "https://huggingface.co/nilc-nlp/glove-300d/resolve/main/"
FONTES = {
    "glove_nilc": {"arquivos": {"embeddings.safetensors": HF + "embeddings.safetensors",
                                "vocab.txt": HF + "vocab.txt"}},
    "fasttext_fb": {"arquivos": {"cc.pt.300.vec.gz":
                                 "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"}},
}


def sem_acento(palavra):
    return "".join(c for c in unicodedata.normalize("NFD", palavra) if unicodedata.category(c) != "Mn")


def palavras_repro():
    """Conjunto de tokens do RePro inteiro (com e sem acento)."""
    dados, mistos = carregar_repro()
    tokens = set()
    for texto in list(dados["texto"]) + list(mistos["texto"]):
        tokens.update(tokenizar(texto))
    return tokens | {sem_acento(t) for t in tokens}


def _progresso(blocos, tam_bloco, total):
    if total > 0 and blocos % 2000 == 0:
        print(f"\r  {blocos * tam_bloco / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)


def baixar(url, destino, forcar=False):
    destino = Path(destino)
    if destino.exists() and not forcar:
        print(f"Já existe: {destino.name}")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url}")
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    urllib.request.urlretrieve(url, tmp, reporthook=_progresso)
    tmp.replace(destino)
    print(f"\n  salvo em {destino} ({destino.stat().st_size / 1e6:,.0f} MB)")
    return destino


def ler_glove_nilc(pasta, manter):
    """Lê safetensors + vocab.txt e devolve (palavras, vetores, rank) filtrados."""
    from safetensors.numpy import load_file
    vetores = load_file(pasta / "embeddings.safetensors")["embeddings"]
    with open(pasta / "vocab.txt", encoding="utf-8") as f:
        vocab = [linha.rstrip("\n") for linha in f]
    assert len(vocab) == len(vetores), (len(vocab), vetores.shape)
    palavras, idx, rank, vistos = [], [], [], set()
    for i, w in enumerate(vocab):
        w = w.lower()
        if w in vistos or not w:
            continue
        if i < TOP_N or w in manter:
            vistos.add(w); palavras.append(w); idx.append(i); rank.append(i)
    return palavras, vetores[idx].astype(np.float32), np.array(rank)


def ler_fasttext_vec(caminho, manter):
    """Lê o .vec.gz linha a linha (2 milhões de palavras) guardando só o necessário.

    O cc.pt.300 diferencia maiúsculas; guardamos a chave em minúsculas e, se a
    mesma palavra aparecer com grafias diferentes ("Bom", "bom"), fica a mais
    frequente (a primeira do arquivo, que é ordenado por frequência).
    """
    palavras, vetores, rank, vistos = [], [], [], set()
    with gzip.open(caminho, "rt", encoding="utf-8", errors="ignore") as f:
        n, dim = map(int, f.readline().split())
        for i, linha in enumerate(f):
            w, _, resto = linha.rstrip().partition(" ")
            w = w.lower()
            if w in vistos or not (i < TOP_N or w in manter):
                continue
            vet = np.array(resto.split(" "), dtype=np.float32)
            if len(vet) != dim:
                continue
            vistos.add(w); palavras.append(w); vetores.append(vet); rank.append(i)
            if i % 200_000 == 0:
                print(f"\r  linha {i:,} de {n:,}", end="", flush=True)
    print()
    return palavras, np.stack(vetores), np.array(rank)


def preparar(nome, manter_brutos=False, forcar=False):
    saida = PASTA_EMB / f"{nome}_filtrado.npz"
    if saida.exists() and not forcar:
        print(f"Arquivo já existe, nada a fazer: {saida}")
        return saida
    pasta = PASTA_BRUTOS / nome
    caminhos = [baixar(url, pasta / arq, forcar) for arq, url in FONTES[nome]["arquivos"].items()]

    print("Filtrando pelo vocabulário do RePro + as", f"{TOP_N:,}", "palavras mais frequentes...")
    manter = palavras_repro()
    if nome == "glove_nilc":
        palavras, vetores, rank = ler_glove_nilc(pasta, manter)
    else:
        palavras, vetores, rank = ler_fasttext_vec(caminhos[0], manter)
    np.savez_compressed(saida, palavras=np.array(palavras), vetores=vetores, rank=rank)
    print(f"  {len(palavras):,} palavras x {vetores.shape[1]} dimensões -> {saida} "
          f"({saida.stat().st_size / 1e6:,.0f} MB)")

    if not manter_brutos:
        for c in caminhos:
            c.unlink()
        pasta.rmdir()
        print(f"  brutos apagados ({pasta})")
    return saida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nomes", nargs="*", metavar="nome",
                        help=f"quais embeddings preparar: {', '.join(FONTES)} (padrão: todos)")
    parser.add_argument("--manter-brutos", action="store_true", help="não apaga os arquivos originais")
    parser.add_argument("--forcar", action="store_true", help="baixa e filtra de novo")
    args = parser.parse_args()
    for nome in args.nomes:
        if nome not in FONTES:
            parser.error(f"embedding desconhecido: {nome} (opções: {', '.join(FONTES)})")
    for nome in args.nomes or list(FONTES):
        preparar(nome, args.manter_brutos, args.forcar)
