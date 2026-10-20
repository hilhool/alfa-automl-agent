"""Собирает ZIP для сабмита: плоский архив без папок.

  python bench/build_zip.py [--out dist/solution.zip]

Платформа копирует в каталог задачи только корневые .py и requirements.txt,
вложенные папки пропадают (инструкция трека). Поэтому agent/X.py кладём в корень
как agent_X.py и переписываем импорты `from agent.X import` в `from agent_X import`.
В репозитории остаётся пакет agent/, переписанный код проверяет стенд: run.py
всегда гоняет именно этот ZIP.

Отказывается собирать, если в архиве папки, CSV, .env или остались импорты
пакета agent. Перед записью каждый файл компилируется.
"""

import argparse
import re
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
MAX_MB = 100
AGENT_IMPORT = re.compile(r"\bfrom agent\.(\w+) import\b")
LEFTOVER = re.compile(r"\bagent\.\w+\b|\bimport agent\b")


def flat_sources() -> dict[str, str]:
    """Имя в архиве -> текст файла."""
    out = {"solution.py": (CODE / "solution.py").read_text(encoding="utf-8")}
    for f in sorted((CODE / "agent").glob("*.py")):
        if f.name != "__init__.py":
            out[f"agent_{f.name}"] = f.read_text(encoding="utf-8")
    for name, text in out.items():
        text = AGENT_IMPORT.sub(lambda m: f"from agent_{m.group(1)} import", text)
        leftover = [line for line in text.splitlines() if LEFTOVER.search(line) and "import" in line]
        if leftover:
            raise SystemExit(f"{name}: импорт пакета agent не переписан: {leftover}")
        compile(text, name, "exec")
        out[name] = text
    out["requirements.txt"] = (CODE / "requirements.txt").read_text(encoding="utf-8")
    return out


def build(out: Path) -> Path:
    sources = flat_sources()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in sources.items():
            zf.writestr(name, text.encode("utf-8"))
    names = zipfile.ZipFile(out).namelist()
    bad = [n for n in names if "/" in n or n.endswith((".csv", ".env")) or "__pycache__" in n]
    if bad or "solution.py" not in names or "requirements.txt" not in names:
        raise SystemExit(f"Плохой состав ZIP: {bad or 'нет solution.py или requirements.txt в корне'}")
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
