import json
import os
import uuid
from datetime import datetime, timezone

import time

from flask import request, render_template, redirect, url_for, flash, session, g, jsonify, Response, stream_with_context
from flask_login import login_user, logout_user, current_user, login_required
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from PIL import Image
from werkzeug.utils import secure_filename

from app import app, db, lm
from app.models import User, Upload, Event
from app.oauth import OAuthSignIn
from app.email import send_magic_link
from app.analytics import log_event, log_event_commit, _session_id, avg_inference_seconds
from app.decorators import admin_required
from app.inference import (
    load_image, detect_corners, classify_corners, detections_to_four_hands,
    hands_to_pbn, hands_to_bbo_url, hand_to_display, draw_detections,
    strip_exif_and_save, ALL_CARDS, SUIT_SYMBOLS,
    inference_slot, queue_position, queue_leave,
)


@lm.user_loader
def load_user(id):
    return db.session.get(User, int(id))


@app.before_request
def before_request():
    g.user = current_user
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()


# --- Main app routes ---

@app.route('/')
def upload_form():
    log_event('page_view', data={'path': '/', 'referrer': request.referrer})
    db.session.commit()
    return render_template('upload.html')


@app.route('/infer', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Generate unique filename to avoid collisions
    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1] or '.jpg'
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)

    # Save uploaded file, strip EXIF for privacy
    pil_img = Image.open(file.stream)
    from PIL import ImageOps
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    strip_exif_and_save(pil_img, file_path)

    # Create a pending upload record so the SSE endpoint can find the file
    training_consent = request.form.get('training_consent') == 'on'
    upload = Upload(
        user_id=current_user.id if current_user.is_authenticated else None,
        original_filename=original_filename,
        stored_filename=stored_filename,
        result_filename='',
        pbn='',
        bbo_url='',
        total_cards=0,
        training_consent=training_consent,
    )
    db.session.add(upload)
    db.session.commit()

    log_event('upload_submitted', upload_id=upload.id,
              data={'training_consent': training_consent})
    db.session.commit()

    return redirect(url_for('processing', upload_id=upload.id))


@app.route('/processing/<int:upload_id>')
def processing(upload_id):
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))
    return render_template('processing.html', upload_id=upload_id)


@app.route('/process/<int:upload_id>')
def process_sse(upload_id):
    """SSE endpoint that runs inference and streams progress."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        return 'Not found', 404

    # Capture what we need before entering the generator
    stored_filename = upload.stored_filename
    ev_session_id = _session_id()
    ev_user_id = current_user.id if current_user.is_authenticated else None

    def generate():
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)

        # Serialize inference per-worker so page loads stay responsive.
        got_slot = inference_slot.acquire(blocking=False)
        queued_at = None
        if not got_slot:
            queued_at = time.time()
            position = queue_position()
            try:
                avg = avg_inference_seconds()
                estimate_s = int((position * avg + 9) // 10) * 10  # round up to nearest 10s
                yield f"data: {json.dumps({'stage': 'queued', 'message': f'Another image is being processed — estimated wait ~{estimate_s}s (position {position})...'})}\n\n"
                inference_slot.acquire()
            finally:
                queue_leave()
        try:
            if queued_at is not None:
                wait_s = time.time() - queued_at
                log_event_commit('inference_queued', upload_id=upload_id,
                                 user_id=ev_user_id, session_id=ev_session_id,
                                 data={'wait_s': round(wait_s, 2)})

            yield f"data: {json.dumps({'stage': 'detecting', 'message': 'Detecting card corners...'})}\n\n"

            t0 = time.time()
            image_np = load_image(file_path)
            corners = detect_corners(image_np)
            t_detect = time.time() - t0

            n_corners = len(corners)
            img_h, img_w = image_np.shape[:2]
            corner_bboxes = [list(c['bbox']) for c in corners]
            yield f"data: {json.dumps({'stage': 'classifying', 'message': f'Classifying {n_corners} corners...', 'time_detect': round(t_detect, 1), 'corners': corner_bboxes, 'img_w': img_w, 'img_h': img_h})}\n\n"

            t0 = time.time()
            detections = classify_corners(image_np, corners)
            t_classify = time.time() - t0

            if not detections:
                os.remove(file_path)
                # Re-fetch upload for DB operations inside generator
                ul = db.session.get(Upload, upload_id)
                if ul:
                    db.session.delete(ul)
                    db.session.commit()
                log_event_commit('inference_failed', upload_id=upload_id,
                                 user_id=ev_user_id, session_id=ev_session_id,
                                 data={'reason': 'no_detections',
                                       'time_detect': round(t_detect, 2),
                                       'time_classify': round(t_classify, 2)})
                yield f"data: {json.dumps({'stage': 'error', 'message': 'No cards were detected in this image. Please try again with a clearer photo, or a different image.'})}\n\n"
                return

            yield f"data: {json.dumps({'stage': 'assigning', 'message': 'Assigning cards to hands...', 'time_classify': round(t_classify, 1)})}\n\n"

            hands, card_positions, inferred_card = detections_to_four_hands(detections)

            pbn = hands_to_pbn(hands)
            bbo_url = hands_to_bbo_url(hands)

            total_cards = sum(len(hands[p][s]) for p in 'nesw' for s in 'SHDC')

            # Draw annotated image
            annotated = draw_detections(image_np, detections, card_positions)
            result_filename = f"{os.path.splitext(stored_filename)[0]}_result.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
            annotated.save(output_path)

            det_records = [
                {'bbox': list(d['bbox']), 'class_name': d['class_name'],
                 'confidence': round(d['confidence'], 3)}
                for d in detections
            ]

            detected_cards = total_cards - 1 if inferred_card else total_cards

            # Re-fetch upload for DB operations inside generator
            ul = db.session.get(Upload, upload_id)
            ul.result_filename = result_filename
            ul.pbn = pbn
            ul.bbo_url = bbo_url
            ul.total_cards = detected_cards
            ul.set_detections(det_records)
            db.session.commit()

            log_event_commit('inference_completed', upload_id=upload_id,
                             user_id=ev_user_id, session_id=ev_session_id,
                             data={'time_detect': round(t_detect, 2),
                                   'time_classify': round(t_classify, 2),
                                   'n_corners': n_corners,
                                   'detected_cards': detected_cards,
                                   'inferred_card': bool(inferred_card)})

            yield f"data: {json.dumps({'stage': 'done', 'redirect': url_for('result', upload_id=upload_id), 'time_classify': round(t_classify, 1)})}\n\n"
        finally:
            inference_slot.release()

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/result/<int:upload_id>')
def result(upload_id):
    """Display result from a completed upload."""
    upload = db.session.get(Upload, upload_id)
    if not upload or not upload.pbn:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    log_event('result_viewed', upload_id=upload_id)
    db.session.commit()

    hands_lists = _parse_pbn_to_hands_lists(upload.pbn)
    display_hands = {p: hand_to_display(hands_lists[p]) for p in 'nesw'}
    hand_counts = {p: sum(len(hands_lists[p][s]) for s in 'SHDC') for p in 'nesw'}
    missing = _compute_missing(hands_lists)

    total_in_pbn = sum(hand_counts[p] for p in 'nesw')
    inferred_card = None
    inferred_card_display = None
    inferred_hand_display = None
    # If we detected 51 and PBN has 52, one was inferred
    if upload.total_cards == 51 and total_in_pbn == 52:
        # Find which card was inferred by comparing detections to PBN
        det_cards = {d['class_name'] for d in (upload.get_detections() or [])}
        pbn_cards = set()
        for p in 'nesw':
            for s in 'SHDC':
                for r in hands_lists[p][s]:
                    pbn_cards.add(f"{r}{s}")
        inferred = pbn_cards - det_cards
        if len(inferred) == 1:
            card_name = inferred.pop()
            # Find which hand it's in
            for p in 'nesw':
                for s in 'SHDC':
                    if card_name[:-1] in hands_lists[p][s] and card_name[-1] == s:
                        rank, suit_letter = card_name[:-1], card_name[-1]
                        suit_symbols = {'S':'\u2660','H':'\u2665','D':'\u2666','C':'\u2663'}
                        hand_names = {'n':'North','e':'East','s':'South','w':'West'}
                        inferred_card = True
                        inferred_card_display = f"{suit_symbols.get(suit_letter, suit_letter)}{rank}"
                        inferred_hand_display = hand_names.get(p, p)
                        break
                if inferred_card:
                    break

    return render_template(
        'result.html',
        hands=display_hands,
        hand_counts=hand_counts,
        total_cards=upload.total_cards,
        pbn=upload.pbn,
        bbo_url=upload.bbo_url,
        image_file=f'output/{upload.result_filename}',
        detections=upload.get_detections() or [],
        upload_id=upload.id,
        missing=missing,
        inferred_card=inferred_card,
        inferred_card_display=inferred_card_display,
        inferred_hand_display=inferred_hand_display,
    )


def _compute_missing(hands):
    """Return missing cards grouped by suit, e.g. {'S': ['A','K'], 'H': [], ...}"""
    found = set()
    for p in 'nesw':
        for s in 'SHDC':
            for r in hands[p][s]:
                found.add(f"{r}{s}")
    missing_cards = ALL_CARDS - found
    by_suit = {'S': [], 'H': [], 'D': [], 'C': []}
    rank_order = {'A':14,'K':13,'Q':12,'J':11,'T':10,'9':9,'8':8,'7':7,'6':6,'5':5,'4':4,'3':3,'2':2}
    for card in missing_cards:
        rank, suit = card[:-1], card[-1]
        by_suit[suit].append(rank)
    for s in by_suit:
        by_suit[s].sort(key=lambda r: rank_order.get(r, 0), reverse=True)
    return by_suit


@app.route('/history')
@login_required
def history():
    uploads = current_user.uploads.all()
    return render_template('history.html', uploads=uploads)


@app.route('/edit/<int:upload_id>')
def edit_hands(upload_id):
    """Simple hand editor — user adjusts cards per hand directly."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    # Parse current PBN into editable hand strings
    hands = _parse_pbn_to_hands(upload.pbn)
    # Convert to hands dict format for _compute_missing
    hands_lists = {p: {s: list(hands[p][s]) for s in 'SHDC'} for p in 'nesw'}
    missing = _compute_missing(hands_lists)

    return render_template('edit_hands.html', upload=upload, hands=hands,
                           image_file=f'output/{upload.result_filename}',
                           original_image_url=url_for('correct_crop_upload', upload_id=upload_id),
                           missing=missing)


@app.route('/edit/<int:upload_id>/save', methods=['POST'])
def save_edit(upload_id):
    """Save manually edited hands and regenerate PBN/BBO."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    from app.inference import hands_to_pbn, hands_to_bbo_url, hand_to_display

    hands = {}
    for player in 'nesw':
        hands[player] = {}
        for suit in 'SHDC':
            raw = request.form.get(f'{player}_{suit}', '').upper()
            ranks = [c for c in raw if c in 'AKQJT98765432']
            hands[player][suit] = sorted(set(ranks),
                key=lambda r: {
                    'A':14,'K':13,'Q':12,'J':11,'T':10,
                    '9':9,'8':8,'7':7,'6':6,'5':5,'4':4,'3':3,'2':2
                }.get(r, 0), reverse=True)

    pbn = hands_to_pbn(hands)
    bbo_url = hands_to_bbo_url(hands)
    edited_total = sum(len(hands[p][s]) for p in 'nesw' for s in 'SHDC')
    detected_cards = upload.total_cards  # original detected count

    upload.pbn = pbn
    upload.bbo_url = bbo_url
    db.session.commit()

    display_hands = {p: hand_to_display(hands[p]) for p in 'nesw'}

    missing = _compute_missing(hands)

    manually_added = max(0, edited_total - detected_cards)

    log_event('edit_saved', upload_id=upload_id,
              data={'manually_added': manually_added,
                    'total_cards': edited_total})
    db.session.commit()

    flash('Hands updated.')
    return render_template(
        'result.html',
        hands=display_hands,
        hand_counts={p: sum(len(hands[p][s]) for s in 'SHDC') for p in 'nesw'},
        total_cards=detected_cards,
        pbn=pbn,
        bbo_url=bbo_url,
        image_file=f'output/{upload.result_filename}',
        detections=[],
        upload_id=upload.id,
        missing=missing,
        manually_added=manually_added,
    )


def _parse_pbn_to_hands(pbn):
    """Parse PBN string into {player: {suit: 'AKQJ...'}} for editing."""
    hands = {p: {'S': '', 'H': '', 'D': '', 'C': ''} for p in 'nesw'}
    try:
        deal = pbn.split('"')[1].split(':')[1].strip().rstrip('"')
        parts = deal.split()
        for i, player in enumerate('nesw'):
            if i < len(parts):
                suits = parts[i].split('.')
                for j, suit in enumerate('SHDC'):
                    if j < len(suits):
                        hands[player][suit] = suits[j]
    except (IndexError, ValueError):
        pass
    return hands


def _parse_pbn_to_hands_lists(pbn):
    """Parse PBN string into {player: {suit: ['A','K',...]}} for display/missing."""
    raw = _parse_pbn_to_hands(pbn)
    return {p: {s: list(raw[p][s]) for s in 'SHDC'} for p in 'nesw'}


@app.route('/correct/<int:upload_id>')
def correct_corners(upload_id):
    """Step 1: Review detected corners, add missed ones, remove false positives."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    log_event('correction_started', upload_id=upload_id)
    db.session.commit()

    detections = upload.get_detections()

    # Split detections into portrait (h > w) and landscape (w >= h) groups
    # to offer two box sizes matching the card orientations in this image
    portrait_w, portrait_h, landscape_w, landscape_h = [], [], [], []
    for d in detections:
        w = d['bbox'][2] - d['bbox'][0]
        h = d['bbox'][3] - d['bbox'][1]
        if h > w:
            portrait_w.append(w)
            portrait_h.append(h)
        else:
            landscape_w.append(w)
            landscape_h.append(h)

    # Build box size options from whichever orientations exist
    box_sizes = []
    if portrait_w:
        box_sizes.append({
            'label': 'Vertical',
            'w': round(sum(portrait_w) / len(portrait_w)),
            'h': round(sum(portrait_h) / len(portrait_h)),
        })
    if landscape_w:
        box_sizes.append({
            'label': 'Horizontal',
            'w': round(sum(landscape_w) / len(landscape_w)),
            'h': round(sum(landscape_h) / len(landscape_h)),
        })
    if not box_sizes:
        box_sizes = [{'label': 'Default', 'w': 40, 'h': 60}]

    # Get original image dimensions for scaling
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.stored_filename)
    from PIL import Image as PILImage
    with PILImage.open(img_path) as img:
        img_w, img_h = img.size

    return render_template(
        'correct_corners.html',
        upload=upload,
        detections_json=json.dumps(detections),
        box_sizes_json=json.dumps(box_sizes),
        img_w=img_w,
        img_h=img_h,
    )


@app.route('/correct/<int:upload_id>/classify', methods=['POST'])
def correct_classify(upload_id):
    """Step 2: Review/correct classification of each corner."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    # corners_json comes from step 1 — the final set of bounding boxes
    corners = json.loads(request.form.get('corners_json', '[]'))

    # Classify any new corners (those without a class_name or with '??')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.stored_filename)
    image_np = load_image(img_path)
    from app.inference import get_classifier, _classify_crop_cnn
    classifier = get_classifier()

    h, w = image_np.shape[:2]
    for corner in corners:
        if corner.get('class_name') and corner['class_name'] != '??':
            continue
        x1, y1, x2, y2 = corner['bbox']
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            corner['class_name'] = '??'
            corner['confidence'] = 0.0
            continue
        crop = image_np[cy1:cy2, cx1:cx2]
        name, conf = _classify_crop_cnn(crop, classifier)
        corner['class_name'] = name
        corner['confidence'] = round(conf, 3)

    # Build card options for dropdowns
    card_options = ['??'] + [f"{r}{s}" for s in 'SHDC'
                             for r in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']]

    return render_template(
        'correct_classify.html',
        upload=upload,
        corners=corners,
        corners_json=json.dumps(corners),
        card_options=card_options,
        img_w=image_np.shape[1],
        img_h=image_np.shape[0],
    )


@app.route('/correct/<int:upload_id>/save', methods=['POST'])
def correct_save(upload_id):
    """Save corrected detections and re-run hand assignment."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        flash('Upload not found.')
        return redirect(url_for('upload_form'))

    corrections = json.loads(request.form.get('corrections_json', '[]'))

    # Filter out '??' entries and build detections for hand assignment
    valid = [c for c in corrections if c.get('class_name') and c['class_name'] != '??']

    # Re-run hand assignment with corrected detections
    fake_detections = []
    for c in valid:
        x1, y1, x2, y2 = c['bbox']
        fake_detections.append({
            'class_name': c['class_name'],
            'confidence': c.get('confidence', 1.0),
            'bbox': (x1, y1, x2, y2),
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
        })

    hands, card_positions, _inferred = detections_to_four_hands(fake_detections)
    pbn = hands_to_pbn(hands)
    bbo_url = hands_to_bbo_url(hands)
    total_cards = sum(len(hands[p][s]) for p in 'nesw' for s in 'SHDC')

    # Draw new annotated image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.stored_filename)
    image_np = load_image(img_path)
    annotated = draw_detections(image_np, fake_detections, card_positions)
    result_filename = f"{os.path.splitext(upload.stored_filename)[0]}_corrected.jpg"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
    annotated.save(output_path)

    # Update upload record
    detected_cards = upload.total_cards  # original detected count
    upload.set_corrections(corrections)
    upload.pbn = pbn
    upload.bbo_url = bbo_url
    upload.result_filename = result_filename
    db.session.commit()

    display_hands = {p: hand_to_display(hands[p]) for p in 'nesw'}
    missing = _compute_missing(hands)

    manually_added = max(0, total_cards - detected_cards)

    log_event('correction_saved', upload_id=upload_id,
              data={'n_corrections': len(corrections),
                    'manually_added': manually_added,
                    'total_cards': total_cards})
    db.session.commit()

    flash('Corrections saved — thank you!')
    return render_template(
        'result.html',
        hands=display_hands,
        hand_counts={p: sum(len(hands[p][s]) for s in 'SHDC') for p in 'nesw'},
        total_cards=detected_cards,
        pbn=pbn,
        bbo_url=bbo_url,
        image_file=f'output/{result_filename}',
        detections=fake_detections,
        upload_id=upload.id,
        missing=missing,
        manually_added=manually_added,
    )


@app.route('/correct/<int:upload_id>/image')
def correct_crop_upload(upload_id):
    """Serve the original uploaded image for the correction canvas."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        return 'Not found', 404
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.stored_filename)
    from flask import send_file
    return send_file(img_path, mimetype='image/jpeg')


@app.route('/correct/<int:upload_id>/crop')
def corner_crop(upload_id):
    """Serve a cropped corner image for the classification review."""
    upload = db.session.get(Upload, upload_id)
    if not upload:
        return 'Not found', 404

    x1 = int(request.args.get('x1', 0))
    y1 = int(request.args.get('y1', 0))
    x2 = int(request.args.get('x2', 0))
    y2 = int(request.args.get('y2', 0))

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.stored_filename)
    image_np = load_image(img_path)
    h, w = image_np.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)

    crop = image_np[cy1:cy2, cx1:cx2]
    from PIL import Image as PILImage
    import io
    pil = PILImage.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, 'JPEG', quality=90)
    buf.seek(0)

    from flask import send_file
    return send_file(buf, mimetype='image/jpeg')


@app.route('/demo')
def demo():
    pbn = '[Deal "N:.AKT984.A.AQT965 T94.532.T972.842 AQJ652.QJ76.KJ3. K873..Q8654.KJ73"]'
    bbo_url = 'https://www.bridgebase.com/tools/handviewer.html?n=SHAKT984DACAQT965&e=ST94H532DT972C842&s=SAQJ652HQJ76DKJ3C&w=SK873HDQ8654CKJ73'
    return render_template('demo.html', pbn=pbn, bbo_url=bbo_url, total_cards=51,
                           inferred_card=True, inferred_card_display='\u26609',
                           inferred_hand_display='East')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# --- Auth routes ---

@app.route('/login')
def login():
    if not current_user.is_anonymous:
        return redirect(url_for('upload_form'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('upload_form'))


@app.route('/magic-link', methods=['POST'])
def magic_link_request():
    if not current_user.is_anonymous:
        return redirect(url_for('upload_form'))
    email = request.form.get('email', '').strip().lower()
    if email:
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        token = s.dumps(email, salt='magic-link')
        send_magic_link(email, token)
        log_event('login_requested', data={'method': 'magic_link'})
        db.session.commit()
    flash('Check your inbox! We\'ve sent a login link to %s.' % email)
    return redirect(url_for('login'))


@app.route('/magic-link/<token>')
def magic_link_verify(token):
    if not current_user.is_anonymous:
        return redirect(url_for('upload_form'))
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = s.loads(token, salt='magic-link', max_age=900)
    except (SignatureExpired, BadSignature):
        flash('That login link is invalid or has expired. Please request a new one.')
        return redirect(url_for('login'))
    user = User.query.filter_by(email=email).first()
    is_new = user is None
    if is_new:
        username = email.split('@')[0]
        unique_username = User.make_unique_username(username)
        user = User(username=unique_username, email=email, display_name=unique_username)
        db.session.add(user)
        db.session.commit()
    login_user(user, True)
    log_event('login_success', user_id=user.id,
              data={'method': 'magic_link', 'is_new': is_new})
    db.session.commit()
    if is_new:
        flash('Welcome to BridgeLens!')
    else:
        flash('You\'re logged in!')
    return redirect(url_for('upload_form'))


@app.route('/authorize/<provider>')
def oauth_authorize(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('upload_form'))
    oauth = OAuthSignIn.get_provider(provider)
    return oauth.authorize()


@app.route('/callback/<provider>')
def oauth_callback(provider):
    if not current_user.is_anonymous:
        return redirect(url_for('upload_form'))
    oauth = OAuthSignIn.get_provider(provider)
    username, email, display_name = oauth.callback()
    if email is None and username is None:
        flash('Authentication failed.')
        return redirect(url_for('upload_form'))
    user = User.query.filter_by(email=email).first()
    is_new = user is None
    if is_new:
        unique_username = User.make_unique_username(username)
        user = User(username=unique_username, email=email, display_name=display_name)
        db.session.add(user)
        db.session.commit()
    login_user(user, True)
    log_event('login_success', user_id=user.id,
              data={'method': provider, 'is_new': is_new})
    db.session.commit()
    if is_new:
        flash('Welcome to BridgeLens!')
    else:
        flash('Welcome back!')
    return redirect(url_for('upload_form'))


# --- Admin ---

def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@app.route('/admin/stats')
@admin_required
def admin_stats():
    from datetime import timedelta
    try:
        days = max(1, min(90, int(request.args.get('days', 7))))
    except ValueError:
        days = 7
    since = datetime.now(timezone.utc) - timedelta(days=days)

    events = Event.query.filter(Event.created_at >= since).all()

    counts_by_type = {}
    sessions_by_type = {}
    for ev in events:
        counts_by_type[ev.event_type] = counts_by_type.get(ev.event_type, 0) + 1
        sessions_by_type.setdefault(ev.event_type, set()).add(ev.session_id)
    unique_sessions_by_type = {k: len(v) for k, v in sessions_by_type.items()}

    funnel_order = [
        ('page_view', 'Visited'),
        ('upload_submitted', 'Uploaded'),
        ('inference_completed', 'Got a result'),
        ('result_viewed', 'Viewed result'),
        ('correction_started', 'Started correction'),
        ('correction_saved', 'Saved correction'),
        ('edit_saved', 'Saved edit'),
    ]
    funnel = [(label, unique_sessions_by_type.get(t, 0), counts_by_type.get(t, 0))
              for t, label in funnel_order]

    inf_times = []
    for ev in events:
        if ev.event_type == 'inference_completed':
            d = ev.get_data()
            t = (d.get('time_detect') or 0) + (d.get('time_classify') or 0)
            if t:
                inf_times.append(t)

    total_inference_attempts = counts_by_type.get('inference_completed', 0) + counts_by_type.get('inference_failed', 0)
    failure_rate = None
    if total_inference_attempts:
        failure_rate = counts_by_type.get('inference_failed', 0) / total_inference_attempts

    logins = [e for e in events if e.event_type == 'login_success']
    new_users = sum(1 for e in logins if (e.get_data() or {}).get('is_new'))

    # Daily timeline (YYYY-MM-DD → count per type)
    timeline = {}
    for ev in events:
        if not ev.created_at:
            continue
        day = ev.created_at.strftime('%Y-%m-%d')
        timeline.setdefault(day, {}).setdefault(ev.event_type, 0)
        timeline[day][ev.event_type] += 1
    timeline_rows = sorted(timeline.items())

    recent = Event.query.filter(Event.created_at >= since).order_by(Event.created_at.desc()).limit(50).all()

    return render_template(
        'admin_stats.html',
        days=days,
        total_events=len(events),
        unique_sessions=len({e.session_id for e in events if e.session_id}),
        unique_users=len({e.user_id for e in events if e.user_id}),
        counts_by_type=counts_by_type,
        funnel=funnel,
        p50_inference=_percentile(inf_times, 0.5),
        p95_inference=_percentile(inf_times, 0.95),
        n_inferences=len(inf_times),
        failure_rate=failure_rate,
        new_users=new_users,
        timeline_rows=timeline_rows,
        recent=recent,
    )
