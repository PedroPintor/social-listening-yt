"""Classificador de sentimento do app: BERTimbau congelado + Regressão Logística.

Escolhido na etapa 5 (ver `docs/notas/decisoes_de_escopo.md`). O BERTimbau não é
ajustado: cada texto vira a média dos vetores da última camada (768 dimensões) e a
Regressão Logística, treinada no RePro por `treinar_classificador.py`, dá a classe.
"""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# Pesos baixados do Hugging Face ficam na pasta do projeto, e não na pasta global do usuário.
# Precisa vir antes de importar o transformers. Apagar a pasta remove os modelos.
os.environ.setdefault("HF_HOME", str(RAIZ / "modelos_baixados"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MODELO_BERT = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 128
CAMINHO_RL = RAIZ / "src-pln" / "models" / "bertimbau_congelado_rl.joblib"


class Vetorizador:
    """Texto -> vetor de 768 dimensões (média dos tokens da última camada do BERTimbau)."""

    def __init__(self, modelo=MODELO_BERT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(modelo)
        self.bert = AutoModel.from_pretrained(modelo).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, textos, batch=32, progresso=None):
        saida = []
        for i in range(0, len(textos), batch):
            lote = [str(t) for t in textos[i:i + batch]]
            enc = self.tok(lote, truncation=True, max_length=MAX_LEN, padding=True,
                           return_tensors="pt").to(self.device)
            h = self.bert(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            saida.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            if progresso:
                progresso(min(i + batch, len(textos)) / len(textos))
        return np.vstack(saida) if saida else np.zeros((0, self.bert.config.hidden_size))


class ClassificadorSentimento:
    """Carrega o BERTimbau e a Regressão Logística salva; classifica textos crus."""

    def __init__(self, caminho_rl=CAMINHO_RL):
        if not Path(caminho_rl).exists():
            raise FileNotFoundError(
                f"Classificador não encontrado em {caminho_rl}. Rode antes: python src/treinar_classificador.py")
        pacote = joblib.load(caminho_rl)
        self.rl, self.info = pacote["rl"], pacote["info"]
        self.vetorizar = Vetorizador(self.info["modelo_bert"])

    def prever(self, textos, progresso=None):
        if len(textos) == 0:
            return np.array([], dtype=object)
        return self.rl.predict(self.vetorizar(list(textos), progresso=progresso))
