"""Treina a Regressão Logística sobre o BERTimbau congelado e salva para o app.

    python src/treinar_classificador.py

Reproduz a configuração do notebook `src-pln/notebooks/05_transformers.ipynb`:
mesmo treino (RePro, sem a validação de 10%), texto original (título + ". " + avaliação),
C = 0,1 (escolhido na validação no notebook) e class_weight="balanced".
Na CPU leva alguns minutos (baixa o BERTimbau, ~450 MB, na primeira vez).
Saída: src-pln/models/bertimbau_congelado_rl.joblib (fora do git).
"""
import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sentimento import CAMINHO_RL, MODELO_BERT, RAIZ, Vetorizador

sys.path.insert(0, str(RAIZ / "src-pln"))
from src.dados import SEED, carregar_repro, dividir_treino_teste, dividir_validacao  # noqa: E402

import joblib  # noqa: E402

C = 0.1

t0 = time.perf_counter()
dados, _ = carregar_repro()
dados["texto_bruto"] = (dados["review_title"].fillna("") + ". " + dados["review_text"].fillna("")).str.strip()
X_train, X_test, y_train, y_test = dividir_treino_teste(dados)
X_tr, _, y_tr, _ = dividir_validacao(X_train, y_train)
T_tr = dados.loc[X_tr.index, "texto_bruto"].tolist()
T_te = dados.loc[X_test.index, "texto_bruto"].tolist()

vetorizar = Vetorizador(MODELO_BERT)
print(f"Gerando vetores de {len(T_tr)} textos de treino e {len(T_te)} de teste ({vetorizar.device})...")
V_tr, V_te = vetorizar(T_tr), vetorizar(T_te)

rl = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED).fit(V_tr, y_tr.values)
f1 = f1_score(y_test.values, rl.predict(V_te), average="macro")
print(f"F1 macro no teste do RePro: {f1:.4f} (notebook 05: 0,8857)")

CAMINHO_RL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"rl": rl, "info": {"modelo_bert": MODELO_BERT, "C": C, "f1_teste_repro": round(f1, 4)}}, CAMINHO_RL)
print(f"Salvo em {CAMINHO_RL} ({(time.perf_counter() - t0) / 60:.1f} min)")
