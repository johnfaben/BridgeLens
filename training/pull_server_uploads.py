"""Pull selected uploads from the production server into the local app.db.

Copies each image into uploads/ and creates a local Upload row carrying the
server's detections, so it can be corrected at /correct/<local_id>. Images
already present locally (matched by md5) are not duplicated; they are just
mapped in the manifest.

Entries go in training_data/manifest.json (see training/manifest.py) under
"server:<id>", with provenance (server user, date, consent, why selected).
New entries default to split 'train'; their own deal_group.

Usage:
    python -m training.pull_server_uploads --tag danny 59 60 61 63
    python -m training.pull_server_uploads --tag consent 5 22 25
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app, db  # noqa: E402
from app.models import Upload  # noqa: E402
from training import manifest as mf  # noqa: E402

SSH_HOST = 'hetzner'
REMOTE_DIR = '/var/www/bridgelens'

REMOTE_DUMP = r'''
import hashlib, json, os, sys
from app import app
from app.models import Upload
ids = [int(x) for x in sys.argv[1:]]
out = []
with app.app_context():
    for u in Upload.query.filter(Upload.id.in_(ids)).order_by(Upload.id):
        path = os.path.join("uploads", u.stored_filename)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
        out.append({
            "server_id": u.id, "user_id": u.user_id,
            "created_at": u.created_at.isoformat(),
            "training_consent": bool(u.training_consent),
            "original_filename": u.original_filename,
            "stored_filename": u.stored_filename,
            "detections_json": u.detections_json,
            "pbn": u.pbn, "bbo_url": u.bbo_url, "total_cards": u.total_cards,
            "md5": md5,
        })
print(json.dumps(out))
'''


def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def fetch_remote_records(ids):
    cmd = ['ssh', '-o', 'BatchMode=yes', SSH_HOST,
           f'cd {REMOTE_DIR} && venv/bin/python - ' + ' '.join(str(i) for i in ids)]
    res = subprocess.run(cmd, input=REMOTE_DUMP, capture_output=True, text=True, check=True)
    # App startup logs go to stderr; the JSON is the last stdout line
    return json.loads(res.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('ids', nargs='+', type=int, help='Server upload ids')
    parser.add_argument('--tag', required=True,
                        help='Why these were selected, e.g. danny, john, consent')
    args = parser.parse_args()

    records = fetch_remote_records(args.ids)
    manifest = mf.load()

    with app.app_context():
        upload_folder = app.config['UPLOAD_FOLDER']
        local_by_md5 = {}
        for u in Upload.query.all():
            p = os.path.join(upload_folder, u.stored_filename)
            if os.path.exists(p):
                local_by_md5.setdefault(md5_file(p), u)

        for rec in records:
            sid = rec['server_id']
            if rec['md5'] is None:
                print(f"[missing] server {sid}: image not on server, skipping")
                continue

            existing = local_by_md5.get(rec['md5'])
            if existing is None:
                local_path = os.path.join(upload_folder, rec['stored_filename'])
                subprocess.run(['scp', '-q', '-o', 'BatchMode=yes',
                                f"{SSH_HOST}:{REMOTE_DIR}/uploads/{rec['stored_filename']}",
                                local_path], check=True)
                upload = Upload(
                    user_id=None,
                    original_filename=rec['original_filename'],
                    stored_filename=rec['stored_filename'],
                    pbn=rec['pbn'],
                    bbo_url=rec['bbo_url'],
                    total_cards=rec['total_cards'],
                    training_consent=rec['training_consent'],
                    detections_json=rec['detections_json'],
                )
                db.session.add(upload)
                db.session.commit()
                local_by_md5[rec['md5']] = upload
                status = 'new'
            else:
                upload = existing
                status = 'exists'

            key = f'server:{sid}'
            previous = manifest.get(key, {})
            manifest[key] = {
                'local_id': upload.id,
                'source': 'server',
                'server_id': sid,
                'kind': previous.get('kind', 'photo'),
                'split': previous.get('split', 'train'),
                'deal_group': previous.get('deal_group', key),
                'tag': args.tag,
                'server_user_id': rec['user_id'],
                'created_at': rec['created_at'],
                'training_consent': rec['training_consent'],
                'stored_filename': upload.stored_filename,
                'md5': rec['md5'],
                'server_total_cards': rec['total_cards'],
            }
            corrected = 'corrected' if upload.corrections_json else 'uncorrected'
            print(f"[{status}] server {sid} -> local {upload.id} "
                  f"({rec['total_cards']} cards, {corrected})")

    mf.save(manifest)
    print(f"Manifest: {mf.MANIFEST_PATH}")


if __name__ == '__main__':
    main()
