"""Score a detector + classifier pair against corrected uploads.

Ground truth is each upload's corrections_json (boxes + labels; '??' = a real
corner too tightly cropped to read). The deal ground truth is the set of
non-'??' card names; hand positions come from running the app's hand
assignment on the corrected corners (no inferred cards). Cards whose corrected
corners are far apart are reported as likely correction slips.

Reports, per photo and in total:
  - detector: corner recall / precision (IoU >= 0.5 against corrected boxes)
  - classifier: accuracy on the corrected (labelled) boxes, confident errors,
    and the most common confusions
  - end to end: cards missing / phantom, cards in the wrong hand, deals exactly
    right, and for every missing card the stage that lost it

Usage:
    python -m training.evaluate                              # live models, holdout photos
    python -m training.evaluate --split train                # live models, training photos
    python -m training.evaluate --classifier runs/x/best.pt --name x
    python -m training.evaluate --ids 29 30 31
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.inference import (  # noqa: E402
    CLASSIFY_THRESHOLD, _classify_crop_cnn, classify_corners, crop_box,
    detections_to_four_hands, get_classifier, load_classifier, spread_cards,
)
from pipeline import _get_corner_model, detect_corners, load_image  # noqa: E402
from training import manifest as mf  # noqa: E402

DB_PATH = os.path.join(ROOT, 'app.db')
EVAL_DIR = os.path.join(ROOT, 'training_data', 'eval')
IOU_MATCH = 0.5
CONFIDENT = 0.8


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes):
    """Greedy one-to-one matching by IoU. Returns {gt_index: pred_index}."""
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt_boxes)
                    for pi, p in enumerate(pred_boxes)), reverse=True)
    matched, used_g, used_p = {}, set(), set()
    for score, gi, pi in pairs:
        if score < IOU_MATCH:
            break
        if gi in used_g or pi in used_p:
            continue
        matched[gi] = pi
        used_g.add(gi)
        used_p.add(pi)
    return matched


def hands_to_positions(hands):
    return {f'{r}{s}': d for d, hand in hands.items() for s, ranks in hand.items() for r in ranks}


def default_ids(conn, split):
    """Corrected real photos in the manifest for one split."""
    ids = mf.photo_ids(split)
    return [i for i in ids
            if conn.execute('SELECT corrections_json IS NOT NULL FROM upload WHERE id=?',
                            (i,)).fetchone()[0]]


def evaluate_upload(row, corner_model, classifier):
    upload_id, stored_filename, corrections_json = row
    image_np = load_image(os.path.join(ROOT, 'uploads', stored_filename))
    gt = json.loads(corrections_json)
    gt_boxes = [g['bbox'] for g in gt]
    gt_labelled = [g for g in gt if g['class_name'] != '??']
    gt_cards = {g['class_name'] for g in gt_labelled}
    gt_positions = {}
    if len(gt_cards) >= 4:
        gt_hands, _, _ = detections_to_four_hands([
            {'class_name': g['class_name'], 'confidence': 1.0,
             'cx': (g['bbox'][0] + g['bbox'][2]) / 2, 'cy': (g['bbox'][1] + g['bbox'][3]) / 2}
            for g in gt_labelled])
        gt_positions = {c: d for c, d in hands_to_positions(gt_hands).items() if c in gt_cards}

    # Likely correction slips: one card's corners spread across the table
    label_warnings = []
    for card in sorted(spread_cards(gt_labelled, image_np.shape)):
        pts = [((g['bbox'][0] + g['bbox'][2]) / 2, (g['bbox'][1] + g['bbox'][3]) / 2)
               for g in gt_labelled if g['class_name'] == card]
        label_warnings.append(f"{card} corners at {', '.join(f'({int(x)},{int(y)})' for x, y in pts)}")

    # Full pipeline as the app runs it
    corners = detect_corners(image_np, corner_model)
    pred_boxes = [c['bbox'] for c in corners]
    # Classify every detected box, keeping the raw label even when the app would drop it
    raw = []
    for c in corners:
        crop, _ = crop_box(image_np, c['bbox'], classifier['crop_expand'])
        raw.append(_classify_crop_cnn(crop, classifier) if crop is not None else ('XX', 0.0))
    detections = classify_corners(image_np, corners, classifier)
    pred_cards = {d['class_name'] for d in detections}
    pred_positions = {}
    if len(pred_cards) >= 4:
        hands, _, _ = detections_to_four_hands(detections)
        pred_positions = hands_to_positions(hands)

    # Detector
    det_match = match_boxes(gt_boxes, pred_boxes)

    # Classifier on the corrected boxes
    cls = {'n': 0, 'correct': 0, 'dropped': 0, 'wrong': 0, 'wrong_confident': 0,
           'unreadable_n': 0, 'unreadable_confident': 0}
    confusions = Counter()
    for g in gt:
        crop, _ = crop_box(image_np, g['bbox'], classifier['crop_expand'])
        if crop is None:
            continue
        name, conf = _classify_crop_cnn(crop, classifier)
        if g['class_name'] == '??':
            cls['unreadable_n'] += 1
            if name != 'XX' and conf >= CONFIDENT:
                cls['unreadable_confident'] += 1
            continue
        cls['n'] += 1
        if name == 'XX' or conf < CLASSIFY_THRESHOLD:
            cls['dropped'] += 1
        elif name == g['class_name']:
            cls['correct'] += 1
        else:
            cls['wrong'] += 1
            cls['wrong_confident'] += conf >= CONFIDENT
            confusions[(g['class_name'], name)] += 1

    # End to end: why each missing card was lost
    lost = []
    for card in sorted(gt_cards - pred_cards):
        idxs = [i for i, g in enumerate(gt) if g['class_name'] == card]
        found = [det_match[i] for i in idxs if i in det_match]
        if not found:
            lost.append({'card': card, 'stage': 'detector'})
            continue
        labels = [raw[p] for p in found]
        read = [(n, round(c, 3)) for n, c in labels if n != 'XX' and c >= CLASSIFY_THRESHOLD]
        if not read:
            lost.append({'card': card, 'stage': 'dropped', 'labels': [(n, round(c, 3)) for n, c in labels]})
        else:
            lost.append({'card': card, 'stage': 'misread', 'labels': read})
    phantom = sorted(pred_cards - gt_cards)
    wrong_hand = sorted(c for c in gt_cards & pred_cards
                        if c in gt_positions and pred_positions.get(c) != gt_positions[c])

    return {
        'upload_id': upload_id,
        'gt_cards': len(gt_cards),
        'pred_cards': len(pred_cards),
        'detector': {'gt_boxes': len(gt_boxes), 'pred_boxes': len(pred_boxes),
                     'matched': len(det_match)},
        'classifier': cls,
        'confusions': [[t, p, n] for (t, p), n in confusions.items()],
        'lost': lost,
        'phantom': phantom,
        'wrong_hand': wrong_hand,
        'card_set_correct': not lost and not phantom,
        'exact': not lost and not phantom and not wrong_hand,
        'label_warnings': label_warnings,
    }


def summarise(results):
    tot = Counter()
    confusions = Counter()
    stages = Counter()
    for r in results:
        tot['gt_boxes'] += r['detector']['gt_boxes']
        tot['pred_boxes'] += r['detector']['pred_boxes']
        tot['matched'] += r['detector']['matched']
        for k, v in r['classifier'].items():
            tot['cls_' + k] += v
        for t, p, n in r['confusions']:
            confusions[(t, p)] += n
        for l in r['lost']:
            stages[l['stage']] += 1
        tot['gt_cards'] += r['gt_cards']
        tot['lost'] += len(r['lost'])
        tot['phantom'] += len(r['phantom'])
        tot['wrong_hand'] += len(r['wrong_hand'])
        tot['exact'] += r['exact']
        tot['card_set_correct'] += r['card_set_correct']

    n = len(results)
    lines = []
    lines.append(f"{'id':>4} {'cards':>9} {'lost':>5} {'phantom':>8} {'wrong hand':>11}  lost cards")
    for r in results:
        lost = ', '.join(f"{l['card']}({l['stage']}" +
                         (f": {'/'.join(x[0] for x in l['labels'])}" if l['stage'] == 'misread' else '') + ')'
                         for l in r['lost'])
        lines.append(f"{r['upload_id']:>4} {r['pred_cards']:>4}/{r['gt_cards']:<4} {len(r['lost']):>5} "
                     f"{len(r['phantom']):>8} {len(r['wrong_hand']):>11}  {lost}")
    lines.append('')
    lines.append(f"Photos: {n}, card set correct: {tot['card_set_correct']}/{n}, "
                 f"deals exactly right (cards and hands): {tot['exact']}/{n}")
    lines.append(f"Cards: {tot['gt_cards']} visible, lost {tot['lost']} "
                 f"({', '.join(f'{k} {v}' for k, v in stages.most_common()) or 'none'}), "
                 f"phantom {tot['phantom']}, wrong hand {tot['wrong_hand']}")
    lines.append(f"Detector: recall {tot['matched']}/{tot['gt_boxes']} "
                 f"({tot['matched'] / max(1, tot['gt_boxes']):.1%}), precision "
                 f"{tot['matched']}/{tot['pred_boxes']} ({tot['matched'] / max(1, tot['pred_boxes']):.1%})")
    lines.append(f"Classifier on corrected boxes: {tot['cls_correct']}/{tot['cls_n']} correct "
                 f"({tot['cls_correct'] / max(1, tot['cls_n']):.1%}), wrong {tot['cls_wrong']} "
                 f"({tot['cls_wrong_confident']} at conf >= {CONFIDENT}), dropped {tot['cls_dropped']}; "
                 f"'??' boxes read confidently as a card: {tot['cls_unreadable_confident']}/{tot['cls_unreadable_n']}")
    if confusions:
        lines.append('Top confusions (true -> predicted): ' +
                     ', '.join(f'{t}->{p} x{c}' for (t, p), c in confusions.most_common(12)))
    warnings = [(r['upload_id'], w) for r in results for w in r['label_warnings']]
    if warnings:
        lines.append('')
        lines.append("Possible correction slips (one card's corners far apart; check and re-correct):")
        lines.extend(f'  #{uid}: {w}' for uid, w in warnings)
    return '\n'.join(lines), dict(tot), dict(stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ids', nargs='+', type=int, help='Local upload ids (overrides --split)')
    parser.add_argument('--split', default='holdout', choices=mf.SPLITS,
                        help='Which corrected photos to score (default: holdout)')
    parser.add_argument('--detector', help='Detector .pt or OpenVINO dir (default: live)')
    parser.add_argument('--classifier', help='Classifier checkpoint (default: live)')
    parser.add_argument('--name', default='live', help='Name for the saved report')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    ids = args.ids or default_ids(conn, args.split)
    if not ids:
        sys.exit(f'No corrected photos in split {args.split!r}')

    if args.detector:
        from ultralytics import YOLO
        corner_model = YOLO(args.detector, task='detect')
    else:
        corner_model = _get_corner_model()
    classifier = load_classifier(args.classifier) if args.classifier else get_classifier()

    results = []
    for upload_id in ids:
        row = conn.execute('SELECT id, stored_filename, corrections_json FROM upload WHERE id=?',
                           (upload_id,)).fetchone()
        if not row or not row[2]:
            print(f"[skip] {upload_id}: not corrected")
            continue
        print(f"[eval] {upload_id}...", flush=True)
        results.append(evaluate_upload(row, corner_model, classifier))

    text, totals, stages = summarise(results)
    print()
    print(text)

    os.makedirs(EVAL_DIR, exist_ok=True)
    out = os.path.join(EVAL_DIR, f'{args.name}.json')
    with open(out, 'w') as f:
        json.dump({'name': args.name, 'split': None if args.ids else args.split,
                   'detector': args.detector or 'live',
                   'classifier': args.classifier or 'live', 'ids': ids,
                   'totals': totals, 'lost_by_stage': stages, 'per_photo': results}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
