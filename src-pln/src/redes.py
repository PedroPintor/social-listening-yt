"""Redes neurais para classificação de sentimentos (etapa 4).

Contém o vocabulário, as três arquiteturas comparadas no notebook
`notebooks/04_rede_neural.ipynb` (BiLSTM, BiGRU e CNN 1D) e funções para
salvar, carregar e usar o modelo treinado. Fica em `src/` para o app poder
carregar o modelo sem depender do notebook.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .dados import limpar, tokenizar

PAD, UNK = "<pad>", "<unk>"


class Vocabulario:
    """Mapeia palavras para índices. Índice 0 = <pad>, 1 = <unk>."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def construir(cls, listas_de_tokens, min_freq=2, extras=()):
        """Palavras com frequência >= min_freq no treino, mais `extras` (opcional)."""
        cont = Counter(t for toks in listas_de_tokens for t in toks)
        itos = [PAD, UNK] + sorted(w for w, c in cont.items() if c >= min_freq)
        vistos = set(itos)
        itos += [w for w in extras if w not in vistos]
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def codificar(self, tokens, max_len):
        """Lista de tokens -> (ids com padding até max_len, tamanho real >= 1)."""
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        n = max(len(ids), 1)
        return ids + [0] * (max_len - len(ids)), n


def codificar_textos(textos, vocab, max_len):
    """Textos já limpos -> tensores (X [N, max_len], tamanhos [N])."""
    pares = [vocab.codificar(tokenizar(t), max_len) for t in textos]
    X = torch.tensor([p[0] for p in pares], dtype=torch.long)
    L = torch.tensor([p[1] for p in pares], dtype=torch.long)
    return X, L


class BiRNN(nn.Module):
    """Embedding -> LSTM ou GRU bidirecional -> max pooling no tempo -> camada linear."""

    def __init__(self, n_vocab, n_classes, emb_dim=300, hidden=128, tipo="lstm",
                 dropout_emb=0.25, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        rnn = nn.LSTM if tipo == "lstm" else nn.GRU
        self.rnn = rnn(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop_emb = nn.Dropout(dropout_emb)
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, tamanhos):
        e = self.drop_emb(self.embedding(x))
        # "pack" faz a direção de trás para frente começar na última palavra real, não no padding
        pacote = pack_padded_sequence(e, tamanhos.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(pacote)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        mascara = (torch.arange(x.size(1), device=x.device)[None, :] < tamanhos[:, None].to(x.device))
        h = h.masked_fill(~mascara[:, :, None], float("-inf"))
        return self.saida(self.drop(h.max(dim=1).values))


class CNNTexto(nn.Module):
    """CNN 1D estilo Kim (2014): filtros de 2, 3, 4 e 5 palavras + max pooling no tempo."""

    def __init__(self, n_vocab, n_classes, emb_dim=300, n_filtros=100, janelas=(2, 3, 4, 5),
                 dropout_emb=0.25, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filtros, k) for k in janelas)
        self.drop_emb = nn.Dropout(dropout_emb)
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(n_filtros * len(janelas), n_classes)

    def forward(self, x, tamanhos=None):
        e = self.drop_emb(self.embedding(x)).transpose(1, 2)          # [B, emb, T]
        h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.saida(self.drop(h))


def criar_modelo(arquitetura, n_vocab, n_classes, **kw):
    if arquitetura in ("bilstm", "bigru"):
        return BiRNN(n_vocab, n_classes, tipo=arquitetura[2:], **kw)
    if arquitetura == "cnn":
        return CNNTexto(n_vocab, n_classes, **kw)
    raise ValueError(arquitetura)


def salvar_modelo(caminho, modelo, vocab, config, classes):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": modelo.state_dict(), "vocab": vocab.itos, "config": config,
                "classes": list(classes)}, caminho)


class ClassificadorSentimento:
    """Carrega o modelo salvo e classifica textos crus (aplica `limpar` e `tokenizar`)."""

    def __init__(self, caminho, device="cpu"):
        pacote = torch.load(caminho, map_location=device, weights_only=False)
        self.config = pacote["config"]
        self.classes = pacote["classes"]
        self.vocab = Vocabulario(pacote["vocab"])
        c = self.config
        self.modelo = criar_modelo(c["arquitetura"], len(self.vocab), len(self.classes),
                                   **c.get("kwargs_modelo", {}))
        self.modelo.load_state_dict(pacote["estado"])
        self.modelo.to(device).eval()
        self.device = device

    @torch.no_grad()
    def prever_proba(self, textos):
        X, L = codificar_textos([limpar(t) for t in textos], self.vocab, self.config["max_len"])
        return torch.softmax(self.modelo(X.to(self.device), L), dim=1).cpu().numpy()

    def prever(self, textos):
        return np.array(self.classes)[self.prever_proba(textos).argmax(axis=1)]
