
#!/usr/bin/env python3
"""Generate trends.json from latest *.xlsx in repo root (no API)."""

from pathlib import Path

import pandas as pd

from trend_engine import DEFAULT_TRENDS_PATH, generate_trends, save_json


def find_latest_xlsx(repo_root: Path) -> Path:
    files = sorted(repo_root.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Keine .xlsx-Datei im Repo-Root gefunden.")
    return files[0]


def main():
    repo_root = Path(".")
    xlsx_path = find_latest_xlsx(repo_root)
    df = pd.read_excel(xlsx_path, sheet_name=0)
    data = generate_trends(df)
    save_json(DEFAULT_TRENDS_PATH, data)
    print(f"trends.json written from {xlsx_path.name}")


if __name__ == "__main__":
    main()
