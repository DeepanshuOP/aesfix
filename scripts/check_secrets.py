"""Secret scanning script for CI and pre-commit checks.

Scans git tracked files for sensitive tokens (HF, Kaggle, Gemini, OpenRouter).
Ensures zero leaks while strictly avoiding printing any secret or prefix.
"""

import re
import subprocess
import sys
from pathlib import Path

RULES = [
    ("HuggingFace Token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("Kaggle Key", re.compile(r"(?i)(?:kaggle|KG)[\-_]?(?:key|token|api|secret)[\s:=]{1,4}['\"]?[0-9a-f]{32}['\"]?")),
    ("Gemini API Key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("OpenRouter API Key", re.compile(r"sk-or-v1-[a-f0-9]{64}|sk-or-[A-Za-z0-9\-_]{20,}")),
]

# Binary / non-text extensions to skip
SKIP_EXTENSIONS = {
    ".pt", ".npz", ".npy", ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".tif", ".tiff", ".bin", ".tar", ".gz", ".zip", ".pyc"
}

def scan_files() -> int:
    try:
        tracked = subprocess.check_output(["git", "ls-files"], text=True, errors="ignore").splitlines()
    except Exception as e:
        print(f"Error reading git files: {e}", file=sys.stderr)
        return 1

    violations = []
    root = Path(__file__).resolve().parent.parent

    for rel_path in tracked:
        p = root / rel_path
        if not p.is_file() or p.suffix.lower() in SKIP_EXTENSIONS:
            continue
        # Skip self or detector rules
        if rel_path in {"scripts/check_secrets.py", ".gitleaks.toml"}:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for rule_name, pattern in RULES:
            if pattern.search(content):
                # Report only rule name and file path; never print secret or length
                violations.append((rule_name, rel_path))

    if violations:
        print(f"Secret scan failed: {len(violations)} violation(s) detected!", file=sys.stderr)
        for rule_name, path in violations:
            print(f"  - [{rule_name}] in {path}", file=sys.stderr)
        return 1

    print("Secret scan clean.")
    return 0

if __name__ == "__main__":
    sys.exit(scan_files())
