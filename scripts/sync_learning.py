"""Sync learning materials from a source directory to the project.

Usage:
    python scripts/sync_learning.py [source_dir]

If no source_dir is given, defaults to ~/Desktop/学习资料
Scans the directory for .ipynb, .md, .pdf, .xlsx, .docx files,
ignores files containing '简历' in the name,
and generates src/data/learning.json + copies files to public/learning/.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = PROJECT_ROOT / "public" / "learning"
DATA_FILE = PROJECT_ROOT / "src" / "data" / "learning.json"

EXTENSIONS = {".ipynb", ".md", ".pdf", ".xlsx", ".docx"}

# Friendly names for known file patterns
NAME_MAP = {
    "week1_textbook.md": ("Week 1 教材", "第一周学习教材，涵盖 Python 核心、并发、元编程、设计模式、PyTorch 基础等"),
    "12周学习计划_修订版.xlsx": ("12周学习计划", "完整的12周AI/ML学习计划总览"),
}

def guess_title_desc(filename: str) -> tuple[str, str]:
    if filename in NAME_MAP:
        return NAME_MAP[filename]

    stem = Path(filename).stem
    # Try to parse W1_Day1_xxx patterns
    parts = stem.replace("-", "_").split("_")
    if len(parts) >= 3 and parts[0].startswith("W") and parts[1].startswith("Day"):
        week = parts[0].upper().replace("W", "W")
        day = parts[1].capitalize()
        topic = " ".join(parts[2:]).replace("_", " ").title()
        title = f"{week} {day} — {topic}"
        desc = f"{week} {day} 学习笔记：{topic}"
        return title, desc

    # Generic fallback
    title = stem.replace("_", " ").replace("-", " ").title()
    return title, f"学习资料：{title}"

def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop" / "学习资料"
    if not source.exists():
        print(f"Source directory not found: {source}")
        sys.exit(1)

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(source.iterdir()):
        if f.suffix.lower() not in EXTENSIONS:
            continue
        if "简历" in f.name:
            continue
        files.append(f)

    print(f"Found {len(files)} files in {source}")

    # Copy files
    for f in files:
        dest = PUBLIC_DIR / f.name
        if not dest.exists() or dest.stat().st_mtime < f.stat().st_mtime:
            shutil.copy2(f, dest)
            print(f"  copied: {f.name}")
        else:
            print(f"  skip (unchanged): {f.name}")

    # Generate learning.json
    entries = []
    for f in files:
        title, desc = guess_title_desc(f.name)
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        tags = []
        stem_upper = f.stem.upper()
        if "W1" in stem_upper or "WEEK1" in stem_upper:
            tags.append("Week 1")
        if "W2" in stem_upper or "WEEK2" in stem_upper:
            tags.append("Week 2")
        if f.suffix == ".ipynb":
            tags.append("Notebook")
        elif f.suffix == ".md":
            tags.append("Markdown")
        elif f.suffix == ".pdf":
            tags.append("PDF")

        entries.append({
            "title": title,
            "description": desc,
            "file": f.name,
            "tags": tags,
            "date": mtime.strftime("%Y-%m-%d"),
        })

    with open(DATA_FILE, "w", encoding="utf-8") as fp:
        json.dump(entries, fp, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(entries)} entries written to {DATA_FILE}")

if __name__ == "__main__":
    main()
