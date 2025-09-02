import os
import re
from pathlib import Path

def rename_json_files(root_dir, dry_run=False):
    # 匹配「数字-数字」，支持多位数字；只替换首次出现
    pattern = re.compile(r'(\d+)-(\d+)')

    for path in Path(root_dir).rglob("*.json"):
        new_name = pattern.sub(r"\1.\2", path.name, count=1)  # 关键：count=1
        if new_name != path.name:
            new_path = path.with_name(new_name)
            if new_path.exists():
                print(f"[SKIP] 目标已存在：{new_path}")
                continue
            print(f"Renaming:\n  {path}\n  -> {new_path}")
            if not dry_run:
                path.rename(new_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python rename_json.py <目录路径> [--dry-run]")
        sys.exit(1)

    root = sys.argv[1]
    dry = (len(sys.argv) == 3 and sys.argv[2] == "--dry-run")
    rename_json_files(root, dry_run=dry)