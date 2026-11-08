"""Import a folder of photos into the local app.db for correction.

Each photo is processed the way the live site processes an upload (EXIF
rotation applied, EXIF stripped, stored under uploads/ with a random name),
run through the live models, and given an Upload row so it can be corrected at
/correct/<local_id>. The manifest entry ("file:<filename>") records the split
and a deal group.

Deal groups: photos taken within --group-gap seconds of the previous one
(EXIF time) are assumed to be the same physical deal. Check the printed groups;
fix any in training_data/manifest.json if needed.

Usage:
    python -m training.import_photos training_data/incoming/batch1 --tag john_phone --split train
    python -m training.import_photos holdout_dir --tag john_phone --split holdout
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime

from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app, db  # noqa: E402
from app.inference import (  # noqa: E402
    classify_corners, detections_to_four_hands, draw_detections, hands_to_bbo_url,
    hands_to_pbn, strip_exif_and_save,
)
from app.models import Upload  # noqa: E402
from pipeline import detect_corners, load_image  # noqa: E402
from training import manifest as mf  # noqa: E402

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306
EXIF_IFD = 0x8769


def taken_at(path):
    exif = Image.open(path).getexif()
    raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    if raw:
        return datetime.strptime(raw, '%Y:%m:%d %H:%M:%S')
    return datetime.fromtimestamp(os.path.getmtime(path))


def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def run_pipeline(stored_filename):
    """Same steps as the live /process endpoint; returns Upload field values."""
    path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    image_np = load_image(path)
    detections = classify_corners(image_np, detect_corners(image_np))
    fields = {'detections_json': json.dumps([
        {'bbox': list(d['bbox']), 'class_name': d['class_name'],
         'confidence': round(d['confidence'], 3)} for d in detections])}
    if len({d['class_name'] for d in detections}) < 4:
        return dict(fields, pbn='', bbo_url='', total_cards=len(detections), result_filename='')
    hands, card_positions, inferred_cards = detections_to_four_hands(detections)
    total = sum(len(hands[p][s]) for p in 'nesw' for s in 'SHDC')
    result_filename = f"{os.path.splitext(stored_filename)[0]}_result.jpg"
    draw_detections(image_np, detections, card_positions).save(
        os.path.join(app.config['OUTPUT_FOLDER'], result_filename))
    return dict(fields, pbn=hands_to_pbn(hands), bbo_url=hands_to_bbo_url(hands),
                total_cards=total - len(inferred_cards),
                result_filename=result_filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folder')
    parser.add_argument('--tag', required=True, help='Who/why, e.g. john_phone')
    parser.add_argument('--split', required=True, choices=mf.SPLITS)
    parser.add_argument('--group-gap', type=int, default=60,
                        help='Max seconds between shots of the same deal (default 60)')
    args = parser.parse_args()

    files = [os.path.join(args.folder, f) for f in os.listdir(args.folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    photos = sorted((taken_at(p), p) for p in files)

    manifest = mf.load()
    known_md5 = {e.get('source_md5') for e in manifest.values()}
    groups = {}
    group, prev_time = None, None

    with app.app_context():
        for when, path in photos:
            name = os.path.basename(path)
            if group is None or (when - prev_time).total_seconds() > args.group_gap:
                group = f'file:{name}'
            prev_time = when
            groups.setdefault(group, []).append(name)

            source_md5 = md5_file(path)
            if source_md5 in known_md5:
                print(f"[exists] {name}")
                continue
            if manifest.get(group, {}).get('split', args.split) != args.split:
                sys.exit(f"{name}: deal group {group} is already in another split")

            img = ImageOps.exif_transpose(Image.open(path))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            ext = os.path.splitext(name)[1].lower() or '.jpg'
            stored_filename = f"{uuid.uuid4().hex}{ext}"
            strip_exif_and_save(img, os.path.join(app.config['UPLOAD_FOLDER'], stored_filename))

            print(f"[import] {name}: running models...", flush=True)
            upload = Upload(user_id=None, original_filename=name, stored_filename=stored_filename,
                            training_consent=True, **run_pipeline(stored_filename))
            db.session.add(upload)
            db.session.commit()

            manifest[f'file:{name}'] = {
                'local_id': upload.id,
                'source': 'file',
                'source_md5': source_md5,
                'tag': args.tag,
                'kind': 'photo',
                'split': args.split,
                'deal_group': group,
                'created_at': when.isoformat(),
                'stored_filename': stored_filename,
                'server_total_cards': upload.total_cards,
            }
            known_md5.add(source_md5)
            mf.save(manifest)
            print(f"         -> local {upload.id}, {upload.total_cards} cards, deal {group}")

    print("\nDeal groups (photos within "
          f"{args.group_gap}s of each other); fix in {mf.MANIFEST_PATH} if wrong:")
    for g, names in groups.items():
        print(f"  {g}: {', '.join(names)}")


if __name__ == '__main__':
    main()
