"""Экспорт ml_api_project/REPORT.md в Word (REPORT.docx).

Зачем:
- Часто в университете требуют отчёт в формате .docx.

Как использовать:
    py -m pip install pypandoc
    py export_report_docx.py

Примечание:
- Если pandoc не установлен, pypandoc попробует скачать его локально.
  Если скачивание запрещено, установите pandoc вручную: https://pandoc.org/installing.html
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

    try:
        pypandoc.get_pandoc_version()
    except OSError:
        pypandoc.download_pandoc()

    pypandoc.convert_file(
        str(md_path),
        to="docx",
        outputfile=str(out_path),
        extra_args=["--from=gfm"],
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("REPORT.docx не создан или пустой")

    print(f"OK: создан {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

