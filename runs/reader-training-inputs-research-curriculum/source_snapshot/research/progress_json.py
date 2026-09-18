"""Atomic JSON progress writes with bounded retries for Windows sharing locks."""
import json
from pathlib import Path
import time


def write_progress(path, data, attempts=20, delay=0.05):
    if attempts < 1:
        raise ValueError('At least one replace attempt is required')
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                # Preserve the full new document in .tmp for explicit recovery.
                raise
            time.sleep(delay)
