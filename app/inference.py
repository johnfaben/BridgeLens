"""Card detection and classification inference logic."""

import math
import os
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

import torch
from torchvision import transforms as T

from config import basedir
from pipeline import detect_corners, load_image

CNN_MODEL_PATH = os.path.join(basedir, 'best_corner_classifier_cnn.pt')
_classifier = None

RANK_ORDER = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
              '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
SUIT_SYMBOLS = {'S': '\u2660', 'H': '\u2665', 'D': '\u2666', 'C': '\u2663'}
ALL_CARDS = {f"{r}{s}" for s in 'SHDC'
             for r in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']}

CLASSIFY_THRESHOLD = 0.30
POSITION_THRESHOLD = 0.50
SPREAD_WARN = 0.2  # fraction of the image diagonal; see spread_cards()

# Serializes inference within a single worker process. Other worker
# processes have their own instance, so global concurrency = num_workers.
inference_slot = threading.Semaphore(1)
_queue_depth_lock = threading.Lock()
_queue_depth = 0  # number of requests waiting on inference_slot


def queue_position():
    """Return 1-based position for a new waiter (1 = next up)."""
    global _queue_depth
    with _queue_depth_lock:
        _queue_depth += 1
        return _queue_depth


def queue_leave():
    global _queue_depth
    with _queue_depth_lock:
        _queue_depth -= 1


def load_classifier(path):
    """Load a CNN classifier checkpoint into the dict used by _classify_crop_cnn."""
    from train_classifier_cnn import CardClassifier
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = CardClassifier(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded CNN classifier ({checkpoint.get('val_acc', 0):.3f} val acc)")
    return {
        'model': model,
        'class_names': checkpoint['class_names'],
        'img_size': checkpoint.get('img_size', 64),
        'crop_expand': checkpoint.get('crop_expand', 1.0),
    }


def get_classifier():
    """Load the live CNN classifier model (cached)."""
    global _classifier
    if _classifier is None:
        _classifier = load_classifier(CNN_MODEL_PATH)
    return _classifier


def _classify_crop_cnn(crop_rgb, classifier_dict):
    """Classify a single crop using the CNN model."""
    img_size = classifier_dict['img_size']
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(crop_rgb).unsqueeze(0)
    with torch.no_grad():
        logits = classifier_dict['model'](tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    class_name = classifier_dict['class_names'][idx.item()]
    return class_name, conf.item()


def crop_box(image_np, bbox, expand=1.0):
    """Clip bbox to the image and return (crop, (x1, y1, x2, y2)), or (None, None) if empty.

    The single crop policy for classification; training data must use it too.
    expand grows the box about its centre (the classifier checkpoint records
    the value it was trained with as crop_expand). The returned box is the
    unexpanded one, clipped.
    """
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = bbox
    if expand != 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) * expand / 2, (y2 - y1) * expand / 2
        ex1, ey1 = max(0, int(cx - hw)), max(0, int(cy - hh))
        ex2, ey2 = min(w, int(cx + hw)), min(h, int(cy + hh))
        box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
        if ex2 <= ex1 or ey2 <= ey1 or box[2] <= box[0] or box[3] <= box[1]:
            return None, None
        return image_np[ey1:ey2, ex1:ex2], box
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(w, int(x2)), min(h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return image_np[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def classify_corners(image_np, corners, classifier=None):
    """Classify each detected corner crop. Returns list of card detections."""
    classifier = classifier or get_classifier()
    detections = []

    for corner in corners:
        crop, box = crop_box(image_np, corner['bbox'], classifier['crop_expand'])
        if crop is None:
            continue
        cx1, cy1, cx2, cy2 = box

        top1_name, top1_conf = _classify_crop_cnn(crop, classifier)

        if top1_conf < CLASSIFY_THRESHOLD or top1_name == 'XX':
            continue

        detections.append({
            'class_name': top1_name,
            'confidence': top1_conf,
            'bbox': (cx1, cy1, cx2, cy2),
            'cx': (cx1 + cx2) / 2,
            'cy': (cy1 + cy2) / 2,
        })

    return detections


def spread_cards(corners, image_shape, threshold=SPREAD_WARN):
    """Card names whose corners are implausibly far apart.

    A card's corners in a fanned hand sit close together, so one label used on
    corners more than `threshold` of the image diagonal apart means at least
    one of those corners is misread.
    """
    diag = math.hypot(image_shape[0], image_shape[1])
    centres = {}
    for c in corners:
        name = c.get('class_name')
        if not name or name in ('??', 'XX'):
            continue
        x1, y1, x2, y2 = c['bbox']
        centres.setdefault(name, []).append(((x1 + x2) / 2, (y1 + y2) / 2))
    return {name for name, pts in centres.items()
            if max((math.dist(p, q) for p in pts for q in pts), default=0) > threshold * diag}


def parse_card(class_name):
    """Parse 'AS', 'TH' into (rank, suit)."""
    return class_name[:-1], class_name[-1]


def _centroids_to_directions(centroids):
    """Map cluster indices to compass directions."""
    direction_map = {}
    direction_map[int(np.argmin(centroids[:, 1]))] = 'n'
    direction_map[int(np.argmax(centroids[:, 1]))] = 's'
    remaining = [i for i in range(4) if i not in direction_map]
    if len(remaining) == 2:
        if centroids[remaining[0], 0] < centroids[remaining[1], 0]:
            direction_map[remaining[0]] = 'w'
            direction_map[remaining[1]] = 'e'
        else:
            direction_map[remaining[0]] = 'e'
            direction_map[remaining[1]] = 'w'
    elif len(remaining) == 1:
        used = set(direction_map.values())
        for d in ['n', 'e', 's', 'w']:
            if d not in used:
                direction_map[remaining[0]] = d
                break
    return direction_map


def detections_to_four_hands(detections):
    """Assign detections to N/E/S/W using KMeans + Hungarian algorithm."""
    card_data = {}
    for d in detections:
        name = d['class_name']
        if name not in card_data:
            card_data[name] = {'xs': [], 'ys': [], 'confs': [], 'best_conf': 0}
        card_data[name]['xs'].append(d['cx'])
        card_data[name]['ys'].append(d['cy'])
        card_data[name]['confs'].append(d['confidence'])
        card_data[name]['best_conf'] = max(card_data[name]['best_conf'], d['confidence'])

    cards = list(card_data.keys())
    positions = np.array([[np.average(card_data[c]['xs'], weights=card_data[c]['confs']),
                           np.average(card_data[c]['ys'], weights=card_data[c]['confs'])]
                          for c in cards])

    card_positions = {c: (positions[i][0], positions[i][1]) for i, c in enumerate(cards)}

    if len(cards) < 4:
        hand = _build_single_hand(cards)
        return {'n': hand, 'e': _empty_hand(), 's': _empty_hand(), 'w': _empty_hand()}, card_positions

    confident_mask = [card_data[c]['best_conf'] >= POSITION_THRESHOLD for c in cards]
    confident_positions = positions[confident_mask] if sum(confident_mask) >= 4 else positions

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(confident_positions)
    centroids = kmeans.cluster_centers_
    direction_map = _centroids_to_directions(centroids)

    n_cards = len(cards)
    dists = np.zeros((n_cards, 4))
    for j in range(4):
        dists[:, j] = np.linalg.norm(positions - centroids[j], axis=1)

    max_per_hand = 13
    n_slots = 4 * max_per_hand

    if n_cards <= n_slots:
        cost_matrix = np.zeros((n_slots, n_slots))
        cost_matrix[:] = 1e9

        for i in range(n_cards):
            for j in range(4):
                for k in range(max_per_hand):
                    cost_matrix[i, j * max_per_hand + k] = dists[i, j]

        for i in range(n_cards, n_slots):
            cost_matrix[i, :] = 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        card_directions = {}
        for i in range(n_cards):
            slot = col_ind[i]
            cluster = slot // max_per_hand
            card_directions[cards[i]] = direction_map.get(cluster, 'n')
    else:
        card_directions = {}
        for i, card in enumerate(cards):
            cluster = int(np.argmin(dists[i]))
            card_directions[card] = direction_map.get(cluster, 'n')

    # If three hands are full, every card not found must be in the fourth.
    # When this is wrong, one of the "full" hands already holds a card that
    # belongs elsewhere, so it never spoils an otherwise correct deal.
    inferred_cards = []
    counts = {d: 0 for d in 'nesw'}
    for d in card_directions.values():
        counts[d] += 1
    short = [d for d, c in counts.items() if c < 13]
    missing = ALL_CARDS - set(cards)
    if len(short) == 1 and len(missing) == 13 - counts[short[0]]:
        for card in sorted(missing):
            card_directions[card] = short[0]
            inferred_cards.append((card, short[0]))

    hands = {d: {'S': [], 'H': [], 'D': [], 'C': []} for d in 'nesw'}
    for name, direction in card_directions.items():
        rank, suit = parse_card(name)
        if suit in hands[direction]:
            hands[direction][suit].append(rank)

    for d in hands:
        for suit in hands[d]:
            hands[d][suit].sort(key=lambda r: RANK_ORDER.get(r, 0), reverse=True)

    return hands, card_positions, inferred_cards


def _build_single_hand(cards):
    hand = {'S': [], 'H': [], 'D': [], 'C': []}
    for name in cards:
        rank, suit = parse_card(name)
        if suit in hand:
            hand[suit].append(rank)
    for suit in hand:
        hand[suit].sort(key=lambda r: RANK_ORDER.get(r, 0), reverse=True)
    return hand


def _empty_hand():
    return {'S': [], 'H': [], 'D': [], 'C': []}


def hand_to_display(hand):
    """Format hand for HTML display."""
    parts = []
    for suit in ['S', 'H', 'D', 'C']:
        ranks = ''.join(hand[suit]) if hand[suit] else '-'
        symbol = SUIT_SYMBOLS[suit]
        parts.append(f'{symbol} {ranks}')
    return '  '.join(parts)


def hands_to_pbn(hands):
    """Generate PBN deal string. Order: N E S W."""
    parts = []
    for player in 'nesw':
        suit_strs = []
        for suit in ['S', 'H', 'D', 'C']:
            ranks = ''.join(hands[player][suit])
            suit_strs.append(ranks)
        parts.append('.'.join(suit_strs))
    return f'[Deal "N:{" ".join(parts)}"]'


def hands_to_bbo_url(hands):
    """Generate BBO handviewer URL for all 4 hands."""
    params = []
    for player in 'nesw':
        hand = hands[player]
        hand_str = (f"S{''.join(hand['S'])}"
                    f"H{''.join(hand['H'])}"
                    f"D{''.join(hand['D'])}"
                    f"C{''.join(hand['C'])}")
        if any(hand[s] for s in 'SHDC'):
            params.append(f"{player}={hand_str}")
    return f"https://www.bridgebase.com/tools/handviewer.html?{'&'.join(params)}"


def draw_detections(image_np, detections, card_positions=None):
    """Draw classified card detections on image. Color by suit."""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    # Scale annotations relative to image size (tuned for ~3000px wide)
    scale = max(img.size) / 3000
    font_path = os.path.join(basedir, 'app', 'static', 'arial.ttf')
    try:
        font = ImageFont.truetype(font_path, max(10, int(35 * scale)))
        font_large = ImageFont.truetype(font_path, max(14, int(50 * scale)))
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_large = font

    box_width = max(1, int(3 * scale))
    dot_r = max(4, int(12 * scale))
    dot_outline = max(1, int(2 * scale))
    text_offset_x = max(8, int(20 * scale))
    text_offset_y = max(10, int(25 * scale))
    stroke_w = max(1, int(4 * scale))

    suit_colors = {'S': '#444444', 'H': '#991111', 'D': '#991111', 'C': '#444444'}

    for d in detections:
        rank, suit = parse_card(d['class_name'])
        color = suit_colors.get(suit, '#ffffff')
        x1, y1, x2, y2 = d['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

    if card_positions:
        for card_name, (cx, cy) in card_positions.items():
            rank, suit = parse_card(card_name)
            color = suit_colors.get(suit, '#ffffff')
            bg = '#ffffff'
            cx, cy = int(cx), int(cy)
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                         fill=color, outline=bg, width=dot_outline)
            draw.text((cx + text_offset_x, cy - text_offset_y), card_name,
                      fill=color, font=font_large, stroke_width=stroke_w,
                      stroke_fill='white')

    return img


def strip_exif_and_save(pil_image, output_path):
    """Save image with EXIF metadata stripped (privacy)."""
    clean = Image.new(pil_image.mode, pil_image.size)
    clean.putdata(list(pil_image.getdata()))
    clean.save(output_path, 'JPEG', quality=90)
