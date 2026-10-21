"""Adversarial validation: насколько test отличается от train и какими колонками.

Классификатор учится отличать строки test от train. AUC около 0,5 значит, что
сдвига нет. Важности признаков показывают, что сдвинуто. Вероятность «похож на
test» для строк train потом пригодится для holdout и весов.
"""

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from agent.core import N_JOBS, prepare

log = logging.getLogger("solution")

MAX_ROWS = 60_000
ADV_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, feature_fraction=0.8,
                  verbose=-1, num_threads=N_JOBS, seed=0)
ROUNDS = 200


@dataclass
class AdversarialReport:
    auc: float
    top: list[tuple[str, float]]       # колонка и доля gain
    train_test_likeness: np.ndarray    # OOF p(test) для строк train


def adversarial(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> AdversarialReport:
    x_tr, x_te = prepare(train, test, features)
    x = pd.concat([x_tr, x_te], ignore_index=True)
    y = np.r_[np.zeros(len(x_tr)), np.ones(len(x_te))]
    rng = np.random.default_rng(0)
    p = np.zeros(len(x))
    # После prepare колонок может стать больше: дата раскладывается на части.
    gain = np.zeros(x.shape[1])
    for fit, val in StratifiedKFold(3, shuffle=True, random_state=0).split(x, y):
        if len(fit) > MAX_ROWS:
            fit = rng.choice(fit, MAX_ROWS, replace=False)
        model = lgb.train(ADV_PARAMS, lgb.Dataset(x.iloc[fit], y[fit]), ROUNDS)
        p[val] = model.predict(x.iloc[val], num_threads=N_JOBS)
        gain += model.feature_importance("gain")
    auc = float(roc_auc_score(y, p))
    share = gain / max(gain.sum(), 1e-12)
    top = sorted(zip(x.columns, share.round(3)), key=lambda t: -t[1])[:5]
    log.info("adversarial: AUC %.3f, сдвинуты сильнее всего %s", auc, top)
    return AdversarialReport(auc, top, p[: len(x_tr)])
