#!/usr/bin/env python3
"""
Simple repository secret scanner (quick pre-commit check).
Searches for common credential patterns and prints file:line:match.
Not a replacement for a full secret management solution, but useful for a quick local check.
"""
import re
import os

ROOT = os.path.dirname(os.path.dirname(__file__)) if os.path.basename(__file__) != __file__ else '.'
PATTERNS = {
    'AWS Access Key ID': re.compile(r'AKIA[0-9A-Z]{16}'),
    'AWS Secret Access Key': re.compile(r'(?i)aws(.{0,20})?secret(.{0,20})?key|[A-Za-z0-9/+=]{40}'),
    'Private Key': re.compile(r'-----BEGIN (RSA |)PRIVATE KEY-----'),
    'SSH Key': re.compile(r'ssh-rsa AAAA[0-9A-Za-z+/]+'),
    'Bearer token': re.compile(r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*'),
    'Generic token (hex)': re.compile(r'\b[0-9a-fA-F]{32,}\b'),
}

EXCLUDE_DIRS = {'.git', 'Model', '__pycache__', '.venv', 'venv', 'node_modules', '.ipynb_checkpoints', '.github'}


def scan_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                for name, pat in PATTERNS.items():
                    if pat.search(line):
                        print(f"{path}:{i}: {name}: {line.strip()}")
    except Exception:
        pass


def walk_and_scan(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        parts = set(os.path.relpath(dirpath, root).split(os.sep))
        if parts & EXCLUDE_DIRS:
            continue
        for fn in filenames:
            if fn.endswith(('.py', '.md', '.txt', '.cfg', '.ini', '.json', '.yaml', '.yml', '.ipynb')):
                scan_file(os.path.join(dirpath, fn))


if __name__ == '__main__':
    print('Running quick secret scan...')
    walk_and_scan('.')
    print('Scan complete. This is a heuristic check only.')
