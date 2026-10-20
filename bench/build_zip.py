"""Собирает ZIP для сабмита: solution.py и agent/*.py, больше ничего.

  python bench/build_zip.py [--out dist/solution.zip]

Отказывается собирать, если в архив попадает CSV, .env или кэш: на платформе
locate может зацепить чужой CSV через rglob. Перед сборкой каждый файл компилируется,
синтаксическая ошибка в модуле иначе всплывёт только на платформе.
"""

import argparse
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
MAX_MB = 128


def payload() -> list[Path]:
    files = [CODE / "solution.py"] + sorted((CODE / "agent").rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


def build(out: Path) -> Path:
    for f in payload():
        compile(f.read_text(encoding="utf-8"), str(f), "exec")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in payload():
            zf.write(f, f.relative_to(CODE).as_posix())
    names = zipfile.ZipFile(out).namelist()
    bad = [n for n in names if n.endswith((".csv", ".env")) or "__pycache__" in n]
    if bad or "solution.py" not in names:
        raise SystemExit(f"Плохой состав ZIP: {bad or 'нет solution.py в корне'}")
    size_mb = out.stat().st_size / 2**20
    if size_mb > MAX_MB:
        raise SystemExit(f"ZIP {size_mb:.1f} МБ, больше {MAX_MB}")
    print(f"{out}: {len(names)} файлов, {size_mb * 1024:.0f} КБ")
    for n in names:
        print("  ", n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=CODE / "dist" / "solution.zip")
    build(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
