"""Offline syntax, JSON, local Markdown link and publication checks. No model calls."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote
ROOT = Path(__file__).resolve().parents[1]
errors = []
counts = {'python': 0, 'json': 0, 'notebook': 0}
for path in ROOT.rglob('*'):
    if not path.is_file() or any(x in path.parts for x in ('.git', '.venv', '__pycache__', '.pytest_cache')):
        continue
    if path.suffix not in ('.py', '.json', '.ipynb', '.md', '.js', '.html', '.css', '.txt', '.sh'):
        continue
    text = path.read_text()
    try:
        if path.suffix == '.py':
            ast.parse(text, filename=str(path)); counts['python'] += 1
        elif path.suffix == '.json':
            json.loads(text); counts['json'] += 1
        elif path.suffix == '.ipynb':
            notebook = json.loads(text); counts['notebook'] += 1
            assert not any(c.get('outputs') for c in notebook['cells']), 'notebook contains outputs'
        elif path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                if not (path.parent / unquote(target.split('#')[0])).exists():
                    errors.append(f'{path.relative_to(ROOT)}: missing link {target}')
    except Exception as exc:
        errors.append(f'{path.relative_to(ROOT)}: {exc}')
    private_host = 'supabase' + '.co'
    user_dir = '/' + 'Users/'
    if private_host in text or user_dir in text:
        errors.append(f'{path.relative_to(ROOT)}: private host/local account path')
    if re.search(r'(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,}', text):
        errors.append(f'{path.relative_to(ROOT)}: possible credential')
print(json.dumps({'counts': counts, 'errors': errors}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
