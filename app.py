"""Flask web app for bridge hand OCR.

Upload a photo of a bridge deal, detect card corners, classify each corner,
assign cards to four hands via KMeans clustering, and output PBN + BBO link.
"""

import os

import numpy as np
from flask import Flask, request, render_template, url_for
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

import json
import torch
from torchvision import transforms as T

from pipeline import detect_corners, load_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'output')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CNN_MODEL_PATH = os.path.join(BASE_DIR, 'best_corner_classifier_cnn.pt')
_classifier = None

RANK_ORDER = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
              '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
SUIT_SYMBOLS = {'S': '\u2660', 'H': '\u2665', 'D': '\u2666', 'C': '\u2663'}
ALL_CARDS = {f"{r}{s}" for s in 'SHDC'
             for r in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']}

CLASSIFY_THRESHOLD = 0.30  # minimum classifier confidence to keep a detection
POSITION_THRESHOLD = 0.50  # minimum confidence for KMeans position assignment


def get_classifier():
    """Load the CNN classifier model."""
    global _classifier
    if _classifier is not None:
        return _classifier

    from train_classifier_cnn import CardClassifier
    checkpoint = torch.load(CNN_MODEL_PATH, map_location='cpu', weights_only=False)
    model = CardClassifier(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _classifier = {
        'model': model,
        'class_names': checkpoint['class_names'],
        'img_size': checkpoint.get('img_size', 64),
    }
    print(f"Loaded CNN classifier ({checkpoint.get('val_acc', 0):.3f} val acc)")

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


def classify_corners(image_np, corners):
    """Classify each detected corner crop. Returns list of card detections."""
    classifier = get_classifier()
    h, w = image_np.shape[:2]
    detections = []

    for corner in corners:
        x1, y1, x2, y2 = corner['bbox']
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = image_np[cy1:cy2, cx1:cx2]
        top1_name, top1_conf = _classify_crop_cnn(crop, classifier)

        # Skip low-confidence and non-card predictions
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


def parse_card(class_name):
    """Parse 'AS', 'TH' into (rank, suit)."""
    return class_name[:-1], class_name[-1]


def _centroids_to_directions(centroids):
    """Map cluster indices to compass directions. Top = N, bottom = S, then W/E by x."""
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
    """Assign detections to N/E/S/W using KMeans + Hungarian algorithm.

    KMeans finds the 4 cluster centroids, then the Hungarian algorithm
    optimally assigns cards to hands with a size constraint of 13 per hand.
    """
    # Average position per unique card, keeping best confidence
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

    # Build card_positions for debug drawing: {card_name: (x, y)}
    card_positions = {c: (positions[i][0], positions[i][1]) for i, c in enumerate(cards)}

    if len(cards) < 4:
        hand = _build_single_hand(cards)
        return {'n': hand, 'e': _empty_hand(), 's': _empty_hand(), 'w': _empty_hand()}, card_positions

    # Use confident detections for initial KMeans fit
    confident_mask = [card_data[c]['best_conf'] >= POSITION_THRESHOLD for c in cards]
    confident_positions = positions[confident_mask] if sum(confident_mask) >= 4 else positions

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(confident_positions)
    centroids = kmeans.cluster_centers_
    direction_map = _centroids_to_directions(centroids)

    # Build cost matrix: distance from each card to each centroid
    # Then use Hungarian algorithm with 13 slots per hand for balanced assignment
    n_cards = len(cards)
    dists = np.zeros((n_cards, 4))
    for j in range(4):
        dists[:, j] = np.linalg.norm(positions - centroids[j], axis=1)

    # Create expanded cost matrix: 13 slots per cluster = 52 columns
    # Each card can go into any of the 13 slots for a given cluster (same cost)
    max_per_hand = 13
    n_slots = 4 * max_per_hand  # 52

    if n_cards <= n_slots:
        # Expand: replicate each cluster's column 13 times
        cost_matrix = np.zeros((n_slots, n_slots))
        cost_matrix[:] = 1e9  # high cost for dummy rows

        for i in range(n_cards):
            for j in range(4):
                for k in range(max_per_hand):
                    cost_matrix[i, j * max_per_hand + k] = dists[i, j]

        # Dummy rows (if fewer than 52 cards) get zero cost to any slot
        for i in range(n_cards, n_slots):
            cost_matrix[i, :] = 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        card_directions = {}
        for i in range(n_cards):
            slot = col_ind[i]
            cluster = slot // max_per_hand
            card_directions[cards[i]] = direction_map.get(cluster, 'n')
    else:
        # More than 52 unique cards (shouldn't happen but fall back to nearest)
        card_directions = {}
        for i, card in enumerate(cards):
            cluster = int(np.argmin(dists[i]))
            card_directions[card] = direction_map.get(cluster, 'n')

    # If 51 cards with 12-13-13-13, infer missing card to short hand
    if len(cards) == 51:
        counts = {}
        for d in card_directions.values():
            counts[d] = counts.get(d, 0) + 1
        if sorted(counts.values()) == [12, 13, 13, 13]:
            missing = ALL_CARDS - set(cards)
            if len(missing) == 1:
                missing_card = missing.pop()
                short_hand = [d for d, c in counts.items() if c == 12][0]
                card_directions[missing_card] = short_hand

    # Build hand dicts
    hands = {d: {'S': [], 'H': [], 'D': [], 'C': []} for d in 'nesw'}
    for name, direction in card_directions.items():
        rank, suit = parse_card(name)
        if suit in hands[direction]:
            hands[direction][suit].append(rank)

    for d in hands:
        for suit in hands[d]:
            hands[d][suit].sort(key=lambda r: RANK_ORDER.get(r, 0), reverse=True)

    return hands, card_positions


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
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_large = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_large = font

    suit_colors = {'S': '#222222', 'H': '#dd0000', 'D': '#dd0000', 'C': '#222222'}

    for d in detections:
        rank, suit = parse_card(d['class_name'])
        color = suit_colors.get(suit, '#ffffff')
        x1, y1, x2, y2 = d['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Draw card position estimates as labeled dots
    if card_positions:
        for card_name, (cx, cy) in card_positions.items():
            rank, suit = parse_card(card_name)
            color = suit_colors.get(suit, '#ffffff')
            bg = '#ffffff'
            r = 6
            cx, cy = int(cx), int(cy)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=bg, width=2)
            draw.text((cx + 10, cy - 10), card_name, fill=color, font=font_large,
                      stroke_width=2, stroke_fill='white')

    return img


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/infer', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Stage 1: Corner detection
    image_np = load_image(file_path)
    corners = detect_corners(image_np)

    # Stage 2: Classify each corner
    detections = classify_corners(image_np, corners)

    # Stage 3: Assign to four hands
    hands, card_positions = detections_to_four_hands(detections)

    # Generate outputs
    pbn = hands_to_pbn(hands)
    bbo_url = hands_to_bbo_url(hands)

    # Count cards per hand
    hand_counts = {}
    total_cards = 0
    for player in 'nesw':
        count = sum(len(hands[player][s]) for s in 'SHDC')
        hand_counts[player] = count
        total_cards += count

    # Draw annotated image
    annotated = draw_detections(image_np, detections, card_positions)
    out_filename = f"{os.path.splitext(filename)[0]}_result.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, out_filename)
    annotated.save(output_path)

    # Format hands for display
    display_hands = {p: hand_to_display(hands[p]) for p in 'nesw'}

    return render_template(
        'result.html',
        hands=display_hands,
        hand_counts=hand_counts,
        total_cards=total_cards,
        pbn=pbn,
        bbo_url=bbo_url,
        image_file=f'output/{out_filename}',
        detections=detections,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
