from functools import lru_cache
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from utils.logging import log_message

# Global cache
_model_cache = {}


@lru_cache(maxsize=1)
def get_yolo_model(detector_model_path):
    """
    Get a cached YOLO model or load a new one.

    Args:
        detector_model_path (str): Path to the YOLO model file

    Returns:
        YOLO: Loaded model instance
    """
    if detector_model_path not in _model_cache:
        _model_cache[detector_model_path] = YOLO(detector_model_path)

    return _model_cache[detector_model_path]


def _shrink_bbox(x1, y1, x2, y2, shrink_ratio=0.05):
    """Shrink bbox slightly inward to avoid text overflow outside bubbles."""
    w = x2 - x1
    h = y2 - y1
    dx = int(w * shrink_ratio)
    dy = int(h * shrink_ratio)
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _merge_overlapping_boxes(boxes, overlap_thresh=0.4):
    """
    Merge only fully overlapping or nested bounding boxes.
    Avoids merging separate but touching/close bubbles.

    Args:
        boxes (list[tuple[int,int,int,int]]): List of (x1,y1,x2,y2)
        overlap_thresh (float): IoU threshold for merging (default 0.4)

    Returns:
        list[tuple[int,int,int,int]]
    """
    if not boxes:
        return []

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            bx1, by1, bx2, by2 = boxes[j]

            # считаем пересечение
            xx1 = max(x1, bx1)
            yy1 = max(y1, by1)
            xx2 = min(x2, bx2)
            yy2 = min(y2, by2)

            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            boxA_area = (x2 - x1) * (y2 - y1)
            boxB_area = (bx2 - bx1) * (by2 - by1)
            union_area = boxA_area + boxB_area - inter_area + 1e-6
            overlap = inter_area / union_area

            # проверка на вложенность (один бокс почти полностью внутри другого)
            is_nested = (
                (bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2) or
                (x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2)
            )

            if overlap > overlap_thresh or is_nested:
                # объединяем только в случае сильного пересечения или вложенности
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)
                used[j] = True

        merged.append((x1, y1, x2, y2))

    return merged



def detect_speech_bubbles(
    image_path: Path,
    detector_model_path,
    confidence: float = 0.35,
    iou: float = 0.2,
    shrink_ratio: float = 0.14,
    merge_overlap: float = 0.01,
    verbose: bool = False,
    device=None
):
    """
    Detect speech bubbles in the given image using the YOLO detection model.

    Args:
        image_path (Path): Path to the input image
        detector_model_path (str): Path to the YOLO detection model
        confidence (float): Confidence threshold for detections
        iou (float): IoU threshold for non-max suppression
        shrink_ratio (float): Percentage to shrink bbox inward
        merge_overlap (float): IoU threshold for merging overlapping boxes
        verbose (bool): Whether to show detailed YOLO output
        device (torch.device | str | int, optional): The device to run the model on.

    Returns:
        list[dict]: List of detection dicts:
            {
              "bbox": (x1, y1, x2, y2),
              "confidence": float,
              "class": str
            }
    """
    detections = []

    _device = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        model = get_yolo_model(detector_model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

    results = model.predict(
        image,
        conf=confidence,
        iou=iou,
        device=_device,
        verbose=verbose
    )

    if not results or len(results) == 0:
        return detections

    result = results[0]
    if result.boxes is None:
        return detections

    raw_boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = _shrink_bbox(x1, y1, x2, y2, shrink_ratio=shrink_ratio)
        raw_boxes.append((x1, y1, x2, y2))

    # объединяем пересекающиеся боксы
    merged_boxes = _merge_overlapping_boxes(raw_boxes, overlap_thresh=merge_overlap)

    for x1, y1, x2, y2 in merged_boxes:
        detection = {
            "bbox": (x1, y1, x2, y2),
            "confidence": 1.0,  # после объединения точной confidence нет
            "class": "bubble"
        }
        detections.append(detection)

    log_message(f"Detector produced {len(detections)} merged bubbles", verbose=verbose)
    return detections
