"""Build classifier and detector training sets from corrected photos.

Only manifest photos in the 'train' split are used; holdout photos are never
exported. Train photos are split train/val by deal group (all shots of one deal
land on the same side), so validation measures new deals, not new angles of
seen ones.

Output, under training_data/datasets/<name>/:

    classifier/{train,val}/<CLASS>/*.jpg
        Crops made with app.inference.crop_box (the live crop policy) at
        --crop-expand. Labelled corners -> their card. Raw detector boxes (the
        live detector is re-run, so boxes the classifier already rejected count
        too) that match no corrected box (false positives) -> XX. '??' corners -> XX or
        dropped (--unreadable). Legacy crops from bridge_ocr are added to train.
    detector/{train,val}/{images,labels}/
        640x640 tiles made with pipeline.tile_image (the live tiling), YOLO
        labels for every corrected box, '??' included. Tiles with no corners
        are kept at --empty-tile-frac. The legacy detector set (bridge_ocr
        corner_detector, ~4.5 GB) is not copied; it lives on the RunPod volume.
    dataset.json
        Settings, counts and the photo ids in each split.

Usage:
    python -m training.export_dataset --name 2026-11-10
    python -m training.export_dataset --name pad115 --crop-expand 1.15 --skip-detector
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import date

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.inference import crop_box  # noqa: E402
from pipeline import CHUNK_SIZE, detect_corners, load_image, tile_image  # noqa: E402
from training import manifest as mf  # noqa: E402
from training.evaluate import iou  # noqa: E402

DB_PATH = os.path.join(ROOT, 'app.db')
DATASETS_DIR = os.path.join(ROOT, 'training_data', 'datasets')
LEGACY_CLASSIFIER = os.path.join(os.path.dirname(ROOT), 'bridge_ocr', 'classifier_data')
# Legacy crops cut at 1.5x by bridge_ocr/label_corners.py; the rest are 1.0x
LEGACY_EXPANDED_PREFIXES = ('PXL_', '1000031637_')
NEGATIVE_IOU = 0.3  # a detection overlapping no corrected box by this much is a false positive

CARDS = [f'{r}{s}' for s in 'CDHS' for r in '23456789TJQKA']
CLASSES = sorted(CARDS) + ['XX']


def in_val(deal_group, val_frac):
    h = int(hashlib.md5(deal_group.encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < val_frac


def save_crop(crop_np, path):
    Image.fromarray(crop_np).save(path, quality=95)


def export_classifier_photo(image_np, upload_id, corrections, detections, out_dir, args, counts):
    def save(crop, cls, tag):
        d = os.path.join(out_dir, cls)
        os.makedirs(d, exist_ok=True)
        save_crop(crop, os.path.join(d, f'u{upload_id}_{tag}.jpg'))
        counts[cls] += 1

    for i, g in enumerate(corrections):
        cls = g['class_name']
        if cls == '??':
            if args.unreadable == 'drop':
                continue
            cls = 'XX'
        crop, _ = crop_box(image_np, g['bbox'], args.crop_expand)
        if crop is not None:
            save(crop, cls, f'c{i}')

    gt_boxes = [g['bbox'] for g in corrections]
    for i, d in enumerate(detections):
        if all(iou(d['bbox'], b) < NEGATIVE_IOU for b in gt_boxes):
            crop, _ = crop_box(image_np, d['bbox'], args.crop_expand)
            if crop is not None:
                save(crop, 'XX', f'fp{i}')


def export_detector_photo(image_np, upload_id, corrections, out_dir, args, counts, rng):
    img_dir, lbl_dir = os.path.join(out_dir, 'images'), os.path.join(out_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    boxes = [g['bbox'] for g in corrections]
    for k, (chunk, x_off, y_off) in enumerate(tile_image(image_np)):
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (x_off <= cx < x_off + CHUNK_SIZE and y_off <= cy < y_off + CHUNK_SIZE):
                continue
            # Clip to the tile, then YOLO-normalise relative to it
            tx1, ty1 = max(x1, x_off) - x_off, max(y1, y_off) - y_off
            tx2, ty2 = min(x2, x_off + CHUNK_SIZE) - x_off, min(y2, y_off + CHUNK_SIZE) - y_off
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            lines.append(f'0 {(tx1 + tx2) / 2 / CHUNK_SIZE:.6f} {(ty1 + ty2) / 2 / CHUNK_SIZE:.6f} '
                         f'{(tx2 - tx1) / CHUNK_SIZE:.6f} {(ty2 - ty1) / CHUNK_SIZE:.6f}')
        if not lines and rng.random() >= args.empty_tile_frac:
            continue
        stem = f'u{upload_id}_t{k}'
        Image.fromarray(chunk).save(os.path.join(img_dir, stem + '.jpg'), quality=95)
        with open(os.path.join(lbl_dir, stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        counts['tiles'] += 1
        counts['boxes'] += len(lines)


def copy_legacy_classifier(src, dst_train, include_expanded, counts):
    """Original (non-augmented) legacy crops, all into train (the legacy val was leaky)."""
    for split in ('train', 'val'):
        split_dir = os.path.join(src, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            if cls not in CLASSES:
                continue
            os.makedirs(os.path.join(dst_train, cls), exist_ok=True)
            for f in os.listdir(os.path.join(split_dir, cls)):
                if '_aug' in f or not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if f.startswith(LEGACY_EXPANDED_PREFIXES) and not include_expanded:
                    continue
                shutil.copy2(os.path.join(split_dir, cls, f),
                             os.path.join(dst_train, cls, f'legacy_{split}_{f}'))
                counts[cls] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--name', default=date.today().isoformat())
    parser.add_argument('--crop-expand', type=float, default=1.0,
                        help='Classifier crop padding; must match what inference will use (recorded in the checkpoint)')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Fraction of train deals held for validation')
    parser.add_argument('--unreadable', choices=('xx', 'drop'), default='xx',
                        help="What '??' corners become for the classifier (default: XX, so it learns to abstain)")
    parser.add_argument('--empty-tile-frac', type=float, default=0.1)
    parser.add_argument('--legacy-classifier', default=LEGACY_CLASSIFIER)
    parser.add_argument('--no-legacy', action='store_true')
    parser.add_argument('--legacy-include-expanded', action='store_true',
                        help='Also copy the 1.5x legacy crops (only sensible with --crop-expand 1.5)')
    parser.add_argument('--stored-negatives', action='store_true',
                        help="XX negatives from the upload's stored detections instead of re-running the detector (faster, far fewer)")
    parser.add_argument('--skip-detector', action='store_true')
    parser.add_argument('--skip-classifier', action='store_true')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing dataset of this name')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = os.path.join(DATASETS_DIR, args.name)
    if os.path.exists(out):
        if not args.force:
            sys.exit(f'{out} exists (use --force to overwrite)')
        shutil.rmtree(out)

    manifest = mf.load()
    conn = sqlite3.connect(DB_PATH)
    entries = {e['local_id']: e for e in manifest.values()}
    rng = random.Random(args.seed)

    splits = {'train': [], 'val': []}
    skipped = []
    for local_id in mf.photo_ids('train', manifest):
        row = conn.execute('SELECT stored_filename, corrections_json, detections_json FROM upload WHERE id=?',
                           (local_id,)).fetchone()
        if not row or not row[1]:
            skipped.append(local_id)
            continue
        split = 'val' if in_val(entries[local_id]['deal_group'], args.val_frac) else 'train'
        splits[split].append(local_id)
    if not splits['val'] and len(splits['train']) > 1:
        sys.exit('No deals landed in val; try a different --val-frac')

    cls_counts = {s: Counter() for s in splits}
    det_counts = {s: Counter() for s in splits}
    for split, ids in splits.items():
        for local_id in ids:
            stored, corr, dets = conn.execute(
                'SELECT stored_filename, corrections_json, detections_json FROM upload WHERE id=?',
                (local_id,)).fetchone()
            image_np = load_image(os.path.join(ROOT, 'uploads', stored))
            corrections = json.loads(corr)
            if args.stored_negatives:
                detections = json.loads(dets) if dets else []
            elif not args.skip_classifier:
                detections = detect_corners(image_np)
            print(f'[{split}] {local_id}', flush=True)
            if not args.skip_classifier:
                export_classifier_photo(image_np, local_id, corrections, detections,
                                        os.path.join(out, 'classifier', split), args, cls_counts[split])
            if not args.skip_detector:
                export_detector_photo(image_np, local_id, corrections,
                                      os.path.join(out, 'detector', split), args, det_counts[split], rng)

    legacy_counts = Counter()
    if not args.skip_classifier:
        # Every class gets a folder in both splits so class indices are fixed
        for split in splits:
            for cls in CLASSES:
                os.makedirs(os.path.join(out, 'classifier', split, cls), exist_ok=True)
        if not args.no_legacy:
            print('Copying legacy classifier crops...', flush=True)
            copy_legacy_classifier(args.legacy_classifier, os.path.join(out, 'classifier', 'train'),
                                   args.legacy_include_expanded, legacy_counts)

    info = {
        'name': args.name,
        'created': date.today().isoformat(),
        'settings': {k: v for k, v in vars(args).items() if k not in ('force',)},
        'classes': CLASSES,
        'photos': splits,
        'skipped_uncorrected': skipped,
        'classifier_counts': {s: dict(c) for s, c in cls_counts.items()},
        'classifier_legacy_counts': dict(legacy_counts),
        'detector_counts': {s: dict(c) for s, c in det_counts.items()},
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'dataset.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nPhotos: train {len(splits['train'])}, val {len(splits['val'])}, "
          f"skipped (uncorrected) {len(skipped)}")
    for s in splits:
        c = cls_counts[s]
        print(f"Classifier {s}: {sum(c.values())} crops ({c['XX']} XX)"
              + (f" + {sum(legacy_counts.values())} legacy" if s == 'train' and legacy_counts else ''))
        print(f"Detector {s}: {det_counts[s]['tiles']} tiles, {det_counts[s]['boxes']} boxes")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
