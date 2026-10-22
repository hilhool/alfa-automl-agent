"""Пробный ZIP для разложения скора платформы на регрессию и классификацию.

Берёт dist/solution.zip и правит solution.py: на задачах не регрессии агент
останавливается после константы (мода). Мода хуже бейзлайна, S_cls = 0, и скор
платформы равен S_reg / 2. В private такая версия не уйдёт: её public ниже.
"""

import zipfile
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist"
MARK = '        log.info("spec: %s", spec)\n'
PROBE = MARK + '        if spec.kind != "regression":\n            return\n'


def main() -> None:
    source, target = DIST / "solution.zip", DIST / "probe_reg_only.zip"
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "solution.py":
                text = data.decode("utf-8")
                assert text.count(MARK) == 1, "не нашли место вставки"
                data = text.replace(MARK, PROBE).encode("utf-8")
            zout.writestr(item, data)
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
