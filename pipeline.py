"""Two-stage card detection pipeline.

Stage 1: Corner detection (best_corner_detector.pt) — implemented.
Stage 2: Card classification from corner crops — TODO: needs a trained classifier model.
"""

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import torchvision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORNER_MODEL_PT = os.path.join(BASE_DIR, 'best_corner_detector.pt')
CORNER_MODEL_OV = os.path.join(BASE_DIR, 'best_corner_detector_openvino_model')

CHUNK_SIZE = 640
OVERLAP = 100
DETECTION_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5

_corner_model = None


def _get_corner_model():
    global _corner_model
    if _corner_model is None:
        if os.path.isdir(CORNER_MODEL_OV):
            _corner_model = YOLO(CORNER_MODEL_OV, task='detect')
            print("Loaded YOLO corner detector (OpenVINO)")
        else:
            _corner_model = YOLO(CORNER_MODEL_PT)
            print("Loaded YOLO corner detector (PyTorch)")
    return _corner_model


def load_image(path):
    """Load an image from file into a numpy array (height, width, 3)."""
    return np.array(Image.open(path).convert('RGB'))


def tile_image(image_np, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split image into overlapping tiles.

    Returns list of (chunk_np, x_offset, y_offset).
    """
    height, width = image_np.shape[:2]
    tiles = []

    if height <= chunk_size and width <= chunk_size:
        if height < chunk_size or width < chunk_size:
            padded = np.zeros((chunk_size, chunk_size, 3), dtype=np.uint8)
            padded[:height, :width] = image_np
            tiles.append((padded, 0, 0))
        else:
            tiles.append((image_np, 0, 0))
        return tiles

    step = chunk_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            y2 = min(y + chunk_size, height)
            x2 = min(x + chunk_size, width)
            y1 = max(0, y2 - chunk_size)
            x1 = max(0, x2 - chunk_size)
            chunk = image_np[y1:y2, x1:x2]

            if chunk.shape[0] < chunk_size or chunk.shape[1] < chunk_size:
                padded = np.zeros((chunk_size, chunk_size, 3), dtype=np.uint8)
                padded[:chunk.shape[0], :chunk.shape[1]] = chunk
                chunk = padded

            tiles.append((chunk, x1, y1))

    return tiles


def detect_corners(image_np):
    """Stage 1: Detect card corners via tiling + NMS dedup.

    Returns list of dicts with keys: bbox (x1,y1,x2,y2), confidence, cx, cy.
    """
    corner_model = _get_corner_model()
    tiles = tile_image(image_np)
    all_boxes = []
    all_scores = []

    for chunk, x_off, y_off in tiles:
        results = corner_model(chunk, verbose=False)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < DETECTION_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_boxes.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off])
                all_scores.append(conf)

    if not all_boxes:
        return []

    # NMS to deduplicate detections from overlapping tiles
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, NMS_IOU_THRESHOLD)

    corners = []
    for i in keep.tolist():
        x1, y1, x2, y2 = all_boxes[i]
        corners.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': all_scores[i],
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
        })
    return corners
