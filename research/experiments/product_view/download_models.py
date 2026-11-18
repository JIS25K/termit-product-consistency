"""Download only pinned official files; default is the small LoRA, --base adds base weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(entry, spec, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = spec.get('sha256')
    if destination.exists() and destination.stat().st_size == spec['size']:
        if not expected or sha256(destination) == expected:
            return
    url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{spec['name']}"
    temporary = destination.with_name(destination.name + '.part')
    for attempt in range(3):
        try:
            print(f"Downloading {spec['name']} ({spec['size'] / 1e6:.1f} MB)", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                while block := response.read(8 * 1024 * 1024):
                    out.write(block)
            if temporary.stat().st_size != spec['size']:
                raise ValueError('Downloaded size does not match pinned metadata')
            if expected and sha256(temporary) != expected:
                raise ValueError('SHA256 mismatch')
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', action='store_true', help='Also download ~58 GB base model')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    downloaded = []
    for key, entry in lock.items():
        for spec in entry['files']:
            name = spec['name']
            if name.startswith('.') or name.endswith(('.gif', '.png', '.jpg', '.mp4')):
                continue
            if key == 'base' and not args.base and name.endswith('.safetensors'):
                continue
            path = ROOT / 'models' / key / name
            download(entry, spec, path)
            downloaded.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                               'sha256': sha256(path), 'repo': entry['repo'], 'revision': entry['revision']})
    (ROOT / 'download_manifest.json').write_text(json.dumps(downloaded, indent=2) + '\n')
    print(f'Verified {len(downloaded)} files.', flush=True)


if __name__ == '__main__':
    main()
