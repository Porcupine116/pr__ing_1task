"""Экспорт REPORT.md в REPORT.docx (Word).

Зачем:
- В университете часто просят отчёт в формате .docx.

Как работает:
- Берём Markdown: ml_assignment/REPORT.md
- Конвертируем в DOCX через библиотеку pypandoc
- Если pandoc не установлен, pypandoc попробует скачать его локально.

Запуск (из корня репозитория):
    py -m pip install pypandoc
    py ml_assignment/export_report_docx.py

Примечание:
- На некоторых машинах скачивание pandoc может быть запрещено политиками сети.
  Тогда установите pandoc вручную: https://pandoc.org/installing.html
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import pypandoc

    root = Path(__file__).resolve().parent
    md_path = root / "REPORT.md"
    out_path = root / "REPORT.docx"

    if not md_path.exists():
        raise FileNotFoundError(f"Не найден файл отчёта: {md_path}")

    # Если pandoc не найден в PATH — попробуем скачать.
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        pypandoc.download_pandoc()

    output = pypandoc.convert_file(
        str(md_path),
        to="docx",
        outputfile=str(out_path),
        extra_args=[
            "--from=gfm",  # GitHub-flavored markdown
        ],
    )

    # pypandoc для outputfile возвращает '' при успехе
    _ = output

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("DOCX не создан или пустой.")

    print(f"OK: создан {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

