"""Baixa o dataset RePro (PROPOR 2024) para a pasta data/.

Uso (a partir de src-pln/):
    python utils/baixar_dados.py            # baixa só se ainda não existir
    python utils/baixar_dados.py --forcar   # baixa de novo e sobrescreve
"""
import argparse
import urllib.request
from pathlib import Path

URL_REPRO = "https://raw.githubusercontent.com/lucasnil/repro/main/RePro.csv"
PASTA_DATA = Path(__file__).resolve().parent.parent / "data"


def baixar_repro(destino=PASTA_DATA / "RePro.csv", url=URL_REPRO, forcar=False):
    """Baixa o RePro.csv para `destino` e devolve o caminho do arquivo."""
    destino = Path(destino)
    if destino.exists() and not forcar:
        print(f"Arquivo já existe, nada a fazer: {destino}")
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url}")
    urllib.request.urlretrieve(url, destino)
    print(f"Salvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forcar", action="store_true", help="baixa mesmo se o arquivo já existir")
    baixar_repro(forcar=parser.parse_args().forcar)
