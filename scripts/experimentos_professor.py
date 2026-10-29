"""Experimentos pedidos na avaliação do professor (27/10/2026).

Gera, para cada configuração e cada modelo, as probabilidades de VALIDAÇÃO e de
TESTE, salvas em arquivos .npz compactos. A análise (tabelas, custo, IC,
calibração, testes estatísticos) é feita depois, a partir desses arquivos.

Etapas (rodadas na ordem de prioridade):
  principal : divisão aleatória (semente 42); colunas {com_saldo, sem_saldo}
              x balanceamento {smote, peso, nenhum}            -> itens 1, 2, 6, 7
  temporal  : divisão por ordem de `step` (60/20/20); smote; {com_saldo, sem_saldo}
                                                              -> item 5
  sementes  : divisão aleatória com sementes 1..4; com_saldo; smote
              (a semente 42 já está na etapa principal)       -> item 4

Uso (na raiz do projeto, com o .venv ativado):
  python scripts/experimentos_professor.py --dataset data/raw/PS_20174392719_1491204439457_log.csv

Opções úteis:
  --etapas principal,temporal,sementes   (padrão: as três, nessa ordem)
  --modelos xgboost,random_forest,logistic_regression
  --saida results/experimentos_professor

O script pode ser interrompido e rodado de novo: configurações já concluídas
são puladas. O progresso fica em <saida>/progresso.log.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data.dataset import load_dataset
from ml.preprocessing.pipeline import build_preprocessing_pipeline

SALDO = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
MODELOS = ["xgboost", "random_forest", "logistic_regression"]


def log(saida: Path, msg: str) -> None:
    linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(linha, flush=True)
    with open(saida / "progresso.log", "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def estimador(nome: str, balanceamento: str, seed: int, y_train: pd.Series):
    """Mesmos hiperparâmetros de ml/models/strategies.py."""
    peso = balanceamento == "peso"
    if nome == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed,
                                  class_weight="balanced" if peso else None)
    if nome == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                      class_weight="balanced" if peso else None)
    if nome == "xgboost":
        from xgboost import XGBClassifier
        spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if peso else 1.0
        return XGBClassifier(random_state=seed, eval_metric="logloss", n_estimators=300,
                             max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=-1, scale_pos_weight=spw)
    raise ValueError(nome)


def pipeline(pre, clf, balanceamento: str, seed: int):
    if balanceamento == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("preprocessor", clone(pre)), ("smote", SMOTE(random_state=seed)),
                            ("classifier", clf)])
    return Pipeline([("preprocessor", clone(pre)), ("classifier", clf)])


def dividir(df: pd.DataFrame, tipo: str, seed: int):
    """Retorna índices (train, val, test). Aleatória: igual ao trainer.py."""
    if tipo == "aleatoria":
        idx = np.arange(len(df)); y = df["target"].to_numpy()
        tr_full, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        tr, va = train_test_split(tr_full, test_size=0.25, random_state=seed, stratify=y[tr_full])
        return tr, va, te
    ordem = np.argsort(df["step"].to_numpy(), kind="stable")
    n = len(ordem); a, b = int(n * 0.6), int(n * 0.8)
    return ordem[:a], ordem[a:b], ordem[b:]


def salvar_split(saida: Path, nome: str, df: pd.DataFrame, va, te) -> None:
    alvo = saida / f"split__{nome}.npz"
    if alvo.exists():
        return
    np.savez_compressed(
        alvo,
        y_val=df["target"].to_numpy()[va].astype(np.int8),
        amount_val=df["amount"].to_numpy()[va].astype(np.float64),
        step_val=df["step"].to_numpy()[va].astype(np.int16),
        y_test=df["target"].to_numpy()[te].astype(np.int8),
        amount_test=df["amount"].to_numpy()[te].astype(np.float64),
        step_test=df["step"].to_numpy()[te].astype(np.int16),
    )


def rodar(saida: Path, df: pd.DataFrame, feature_cols: list[str], etapa: str, divisao: str,
          seed: int, colunas: str, balanceamento: str, modelos: list[str]) -> None:
    split_nome = f"{divisao}_s{seed}"
    tr, va, te = dividir(df, divisao, seed)
    salvar_split(saida, split_nome, df, va, te)
    cols = [c for c in feature_cols if not (colunas == "sem_saldo" and c in SALDO)]
    X = df[cols]; y = df["target"]
    pre, _ = build_preprocessing_pipeline(X.iloc[tr], max_categorical_cardinality=1000)
    for m in modelos:
        alvo = saida / f"probs__{etapa}__{split_nome}__{colunas}__{balanceamento}__{m}.npz"
        if alvo.exists():
            log(saida, f"pulado (já existe): {alvo.name}")
            continue
        t0 = time.time()
        p1 = pipeline(pre, estimador(m, balanceamento, seed, y.iloc[tr]), balanceamento, seed)
        p1.fit(X.iloc[tr], y.iloc[tr])
        prob_val = p1.predict_proba(X.iloc[va])[:, 1]
        tr_full = np.concatenate([tr, va])
        p2 = pipeline(pre, estimador(m, balanceamento, seed, y.iloc[tr_full]), balanceamento, seed)
        p2.fit(X.iloc[tr_full], y.iloc[tr_full])
        prob_test = p2.predict_proba(X.iloc[te])[:, 1]
        np.savez_compressed(alvo, prob_val=prob_val.astype(np.float32),
                            prob_test=prob_test.astype(np.float32),
                            features=np.array(cols))
        log(saida, f"ok: {alvo.name} ({(time.time() - t0) / 60:.1f} min)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--saida", default=str(ROOT / "results" / "experimentos_professor"))
    ap.add_argument("--etapas", default="principal,temporal,sementes")
    ap.add_argument("--modelos", default=",".join(MODELOS))
    args = ap.parse_args()
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]

    versoes = {"python": platform.python_version(), "sistema": platform.platform()}
    for pkg in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]:
        try:
            versoes[pkg] = version(pkg)
        except PackageNotFoundError:
            versoes[pkg] = None
    (saida / "versoes.json").write_text(json.dumps(versoes, indent=2), encoding="utf-8")

    ds = load_dataset(args.dataset)
    df = ds.frame.reset_index(drop=True)
    feature_cols = ds.feature_columns
    log(saida, f"dataset: {len(df)} linhas, {int(df['target'].sum())} fraudes; versões: {versoes}")

    plano = []
    etapas = [e.strip() for e in args.etapas.split(",")]
    if "principal" in etapas:
        for colunas in ["com_saldo", "sem_saldo"]:
            for bal in ["smote", "peso", "nenhum"]:
                plano.append(("principal", "aleatoria", 42, colunas, bal))
    if "temporal" in etapas:
        for colunas in ["com_saldo", "sem_saldo"]:
            plano.append(("temporal", "temporal", 42, colunas, "smote"))
    if "sementes" in etapas:
        for seed in [1, 2, 3, 4]:
            plano.append(("sementes", "aleatoria", seed, "com_saldo", "smote"))

    for i, (etapa, divisao, seed, colunas, bal) in enumerate(plano, 1):
        log(saida, f"[{i}/{len(plano)}] {etapa} | divisão={divisao} | semente={seed} | {colunas} | {bal}")
        rodar(saida, df, feature_cols, etapa, divisao, seed, colunas, bal, modelos)
    log(saida, "FIM. Compacte a pasta de saída e envie.")


if __name__ == "__main__":
    main()
