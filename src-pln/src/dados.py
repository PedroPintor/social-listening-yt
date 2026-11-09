"""Preparação do RePro para a classificação de sentimentos.

Reproduz exatamente os passos do notebook da etapa 2
(`notebooks/02_etapa_analise_sentimentos.ipynb`), para que todas as etapas
usem os mesmos textos e o mesmo conjunto de teste:

1. `ast.literal_eval` na coluna `polarity` (é uma lista em texto);
2. rótulo com 3 classes (positivo, negativo, neutro); avaliações com mais de
   um sentimento viram "misto" e ficam fora do treino e do teste;
3. texto = título + ". " + texto da avaliação, passado pela função `limpar`;
4. `train_test_split(test_size=0.2, stratify=sentimento, random_state=42)`.

Uso (a partir de um notebook em src-pln/notebooks/):

    import sys; sys.path.insert(0, "..")
    from src.dados import carregar_repro, dividir_treino_teste
    dados, mistos = carregar_repro()
    X_train, X_test, y_train, y_test = dividir_treino_teste(dados)
"""
import ast
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CAMINHO_REPRO = Path(__file__).resolve().parent.parent / "data" / "RePro.csv"
CLASSES = ["positivo", "neutro", "negativo"]


def rotulo(pol):
    """Lista de polaridades -> 'positivo' | 'negativo' | 'neutro' | 'misto'."""
    if len(pol) > 1:
        return "misto"
    return pol[0].lower()


def limpar(texto):
    """Mesma limpeza da etapa 2: minúsculas, sem links e sem números."""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)   # links
    texto = re.sub(r"\d+", " ", texto)                   # números
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def carregar_repro(caminho=CAMINHO_REPRO):
    """Lê o RePro e devolve (dados, mistos).

    `dados` tem só as avaliações de um sentimento (3 classes) e `mistos`, as
    avaliações com mais de um sentimento. As duas tabelas têm as colunas
    originais mais `sentimento` e `texto` (título + ". " + texto, já limpos).
    """
    df = pd.read_csv(caminho)
    df["polarity_list"] = df["polarity"].apply(ast.literal_eval)
    df["sentimento"] = df["polarity_list"].apply(rotulo)
    df["texto"] = (df["review_title"].fillna("") + ". " + df["review_text"].fillna("")).apply(limpar)
    dados = df[df["sentimento"] != "misto"].copy()
    mistos = df[df["sentimento"] == "misto"].copy()
    return dados, mistos


def dividir_treino_teste(dados, test_size=0.2, seed=SEED):
    """Divisão 80/20 estratificada, idêntica à da etapa 2.

    Devolve X_train, X_test, y_train, y_test (Series do pandas).
    """
    return train_test_split(dados["texto"], dados["sentimento"], test_size=test_size,
                            stratify=dados["sentimento"], random_state=seed)


def dividir_validacao(X_train, y_train, val_size=0.1, seed=SEED):
    """Separa uma validação estratificada de dentro do treino (para o early stopping).

    O teste não é tocado. Devolve X_tr, X_val, y_tr, y_val.
    """
    return train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)


# --- Tokenização simples para as redes neurais (etapa 4) ---------------------
# Palavras (com acento e hífen interno, como "guarda-roupa") ou um símbolo
# isolado (pontuação, emoji). O texto já chega em minúsculas pelo `limpar`.
_RE_TOKEN = re.compile(r"\w+(?:-\w+)*|[^\w\s]")


def tokenizar(texto):
    """'Não gostei!! 🙄' -> ['não', 'gostei', '!', '!', '🙄'] (minúsculas, acentos mantidos)."""
    return _RE_TOKEN.findall(str(texto).lower())
