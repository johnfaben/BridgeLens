"""The training-data manifest: which local uploads are training/holdout photos.

training_data/manifest.json maps a source key to an entry:

    "server:63"            pulled from production upload 63 (pull_server_uploads.py)
    "file:PXL_....jpg"     imported from a local photo (import_photos.py)

Each entry has at least:
    local_id    Upload.id in the local app.db
    source      'server' | 'file'
    tag         who/why, e.g. danny, john, consent, john_phone
    kind        'photo' | 'online_screenshot' | 'printed_diagram' (non-photos are excluded)
    split       'train' | 'holdout' (holdout photos are never trained on)
    deal_group  photos of the same physical deal share this; keep a group in one split
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, 'training_data', 'manifest.json')
EXCLUDED_KINDS = {'printed_diagram', 'online_screenshot'}
SPLITS = ('train', 'holdout')


def load():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def by_local_id(manifest=None):
    """local upload id -> entry (with its manifest key added as 'key')."""
    manifest = load() if manifest is None else manifest
    return {e['local_id']: dict(e, key=k) for k, e in manifest.items()}


def photo_ids(split=None, manifest=None):
    """Local ids of real photos, optionally restricted to one split."""
    manifest = load() if manifest is None else manifest
    return sorted(e['local_id'] for e in manifest.values()
                  if e.get('kind', 'photo') not in EXCLUDED_KINDS
                  and (split is None or e.get('split') == split))
