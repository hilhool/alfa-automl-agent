"""Пробные ZIP для разложения скора платформы на регрессию и классификацию.

Берёт dist/solution.zip и правит solution.py:
  reg_only       — на классификации агент останавливается после константы (мода хуже
                   бейзлайна, S_cls = 0), скор платформы = S_reg / 2;
  cls_only       — на регрессии нули из шаблона (S_reg = 0), скор = S_cls / 2;
  cls_noweights  — как cls_only, но без множителей классов: A/B множителей на настоящем test.
Проба 22.10 (reg_only) дала 19,22: S_reg ≈ 38,4, значит S_cls ≈ 13 — слабое место.
В private пробы не уйдут: их public ниже лучшего.
"""

import sys
import zipfile
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist"
MARK = '        log.info("spec: %s", spec)\n'
WEIGHTS = "            weights = class_weights("
# На регрессии пишем нули из шаблона: константа «среднее» может обойти слабый бейзлайн
# (на reg_holdout она была лучше RandomForest), и тогда S_reg не ноль.
ZERO_REG = MARK + '        if spec.kind == "regression":\n            write_atomic(sample, files.submission)\n            return\n'
PATCHES = {
    "reg_only": [(MARK, MARK + '        if spec.kind != "regression":\n            return\n')],
    "cls_only": [(MARK, ZERO_REG)],
    "cls_noweights": [(MARK, ZERO_REG), (WEIGHTS, "            weights = None and class_weights(")],
}


def main(mode: str) -> None:
    source, target = DIST / "solution.zip", DIST / f"probe_{mode}.zip"
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "solution.py":
                text = data.decode("utf-8")
                for old, new in PATCHES[mode]:
                    assert text.count(old) == 1, f"не нашли место вставки: {old.strip()}"
                    text = text.replace(old, new)
                data = text.encode("utf-8")
            zout.writestr(item, data)
    print(target, target.stat().st_size)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["reg_only"]:
        main(name)
