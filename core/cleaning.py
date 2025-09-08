import cv2
import largestinteriorrectangle
import numpy as np
from PIL import Image

from utils.logging import log_message
from .image_utils import pil_to_cv2

# -----------------------------
# Вспомогательные функции iou
# -----------------------------

def _ensure_ultralytics_loaded():
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except Exception:
        return False


def _bbox_iou(box_a, box_b):
    """
    IoU двух боксов в формате (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _yolo_seg_infer(image_bgr, model_path, confidence=0.35, device=None, verbose=False):
    """
    Запуск YOLOv8-seg и парсинг результатов в список сегментов:
    [
        {
            "bbox": (x1, y1, x2, y2),
            "mask_points": [(x,y), ...],  # координаты полигона
            "conf": float
        }, ...
    ]
    """
    if not _ensure_ultralytics_loaded():
        raise RuntimeError(
            "Ultralytics/YOLO не установлен. Установите пакет 'ultralytics' для работы клинера."
        )
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Важно: запускаем один раз на всю страницу, чтобы потом сопоставлять с детектами
    results = model.predict(source=image_bgr, conf=confidence, device=device, verbose=False)
    segments = []
    for res in results:
        # res.boxes.xyxy (N,4), res.boxes.conf (N,), res.masks.xy список полигонов (N списков)
        if res.masks is None or res.boxes is None:
            continue

        xyxys = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        polys = res.masks.xy  # список из N элементов; каждый — np.ndarray (K,2) в float

        for i in range(len(polys)):
            poly = polys[i]
            if poly is None or len(poly) < 3:
                continue
            x1, y1, x2, y2 = xyxys[i]
            conf = float(confs[i]) if i < len(confs) else 0.0
            # приводим к int для fillPoly
            mask_points = [(float(x), float(y)) for x, y in poly]
            segments.append({
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "mask_points": mask_points,
                "conf": conf
            })

    log_message(f"Cleaner YOLO produced {len(segments)} segments", verbose=verbose)
    return segments


def _attach_masks_to_detections(detections, segs, iou_threshold=0.3, verbose=False):
    """
    На каждый детект (bbox) вешаем ближайшую (по IoU bbox-ов) сегментацию.
    Если несколько кандидатов — берём с максимальным IoU.
    Возвращаем новый список детектов с полем "mask_points" при наличии совпадений.
    """
    new_dets = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det_box = (float(x1), float(y1), float(x2), float(y2))
        best_seg = None
        best_iou = 0.0
        for s in segs:
            seg_box = s["bbox"]
            iou = _bbox_iou(det_box, seg_box)
            if iou > best_iou:
                best_iou = iou
                best_seg = s
        det_copy = dict(det)
        if best_seg is not None and best_iou >= iou_threshold:
            det_copy["mask_points"] = best_seg["mask_points"]
            det_copy["cleaner_iou"] = best_iou
        else:
            log_message(
                f"Cleaner: no matching mask for detection {det['bbox']} (best IoU={best_iou:.2f})",
                verbose=verbose
            )
        new_dets.append(det_copy)
    return new_dets


def _detections_have_masks(detections):
    return any(("mask_points" in d and d["mask_points"]) for d in detections)


# -----------------------------
# Основная функция
# -----------------------------

def clean_speech_bubbles(
    image_path,
    cleaner_model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    dilation_kernel_size=7,
    dilation_iterations=1,
    use_otsu_threshold: bool = False,
    min_contour_area=50,
    closing_kernel_size=7,
    closing_iterations=1,
    closing_kernel_shape=cv2.MORPH_ELLIPSE,
    constraint_erosion_kernel_size=5,
    constraint_erosion_iterations=1,
    verbose: bool = False,
):
    """
    Clean speech bubbles in the given image using the segmentation-based cleaner model
    (YOLOv8-seg) and refined masking.

    РОЛИ:
    - Детектор (ОТДЕЛЬНО, в другом модуле): выдаёт bbox баблов.
    - Клинер (ЗДЕСЬ, по cleaner_model_path): даёт сегментационные маски баблов, мы по ним
      строим итоговую маску интерьера и заливаем.

    Args:
        image_path (str): путь к изображению страницы.
        cleaner_model_path (str): путь к YOLOv8-seg модели очистки (/models/yolov8m_seg-speech-bubble.pt).
        confidence (float): порог для клинера.
        pre_computed_detections (list[dict] | None): детекты из детектора. Каждый dict минимум с "bbox"=(x1,y1,x2,y2).
            - Если уже есть "mask_points", они будут использованы напрямую.
            - Если масок нет, клинер запустится и сопоставит свои сегменты с bbox.
        device: устройство для YOLO ("cuda", "cpu", int и т.д.)
        ... (остальные параметры управляют пост-обработкой маски)

    Returns:
        cleaned_image (np.ndarray BGR):
        processed_bubbles (list[dict]): по каждому баблу:
            {
              "mask": np.ndarray (uint8, 0/255),
              "color": (B, G, R),
              "bbox": (x1,y1,x2,y2),
              "lir_bbox": [x,y,w,h] | None
            }

    Raises:
        ValueError: если не удалось загрузить изображение.
        RuntimeError: в случае ошибок обработки.
    """
    try:
        pil_image = Image.open(image_path)
        image = pil_to_cv2(pil_image)
        img_height, img_width = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cleaned_image = image.copy()

        # --------------------------
        # Получаем детекты + маски
        # --------------------------
        detections = pre_computed_detections if pre_computed_detections is not None else []

        # Если вообще не дали детектов — аварийный фолбэк: используем клинер как детектор сегментов.
        if not detections:
            log_message(
                "WARN: No pre_computed_detections passed. Falling back to cleaner-only segmentation "
                "(detection responsibility ideally should be separate).",
                always_print=True
            )
            segs = _yolo_seg_infer(image, cleaner_model_path, confidence=confidence, device=device, verbose=verbose)
            # Преобразуем сегментации в формат детектов
            detections = [{"bbox": s["bbox"], "mask_points": s["mask_points"], "conf": s.get("conf", 0.0)} for s in segs]

        # Если детекты есть, но у части/всех нет масок — подтянем их клинером и сопоставим.
        if not _detections_have_masks(detections):
            segs = _yolo_seg_infer(image, cleaner_model_path, confidence=confidence, device=device, verbose=verbose)
            detections = _attach_masks_to_detections(detections, segs, iou_threshold=0.05, verbose=verbose)

        # --------------------------
        # Пост-обработка и заливка
        # --------------------------
        processed_bubbles = []
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        closing_kernel = cv2.getStructuringElement(closing_kernel_shape, (closing_kernel_size, closing_kernel_size))
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (constraint_erosion_kernel_size, constraint_erosion_kernel_size)
        )

        for detection in detections:
            if "mask_points" not in detection or not detection["mask_points"]:
                log_message(f"Skipping detection without mask points: {detection.get('bbox')}", verbose=verbose)
                continue

            try:
                points_list = detection["mask_points"]
                points = np.array(points_list, dtype=np.float32)

                if len(points.shape) == 3 and points.shape[1] == 1:
                    points_int = np.round(points).astype(int)
                elif len(points.shape) == 2 and points.shape[1] == 2:
                    points_int = np.round(points).astype(int).reshape((-1, 1, 2))
                else:
                    log_message(
                        f"Unexpected mask points format for detection {detection.get('bbox')}. Skipping.",
                        verbose=verbose,
                    )
                    continue

                original_yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(original_yolo_mask, [points_int], 255)

                # ROI: слегка расширяем, чтобы захватить края текста
                roi_mask = cv2.dilate(original_yolo_mask, dilation_kernel, iterations=dilation_iterations)

                masked_pixels = img_gray[original_yolo_mask == 255]
                if masked_pixels.size == 0:
                    log_message(
                        f"Skipping detection {detection.get('bbox')} due to empty mask after indexing.", verbose=verbose
                    )
                    continue

                mean_pixel_value = np.mean(masked_pixels)
                is_black_bubble = mean_pixel_value < 128
                fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
                log_message(
                    f"Detection {detection.get('bbox')}: Mean={mean_pixel_value:.1f} -> "
                    f"{'Black' if is_black_bubble else 'White'} Bubble. Fill: {fill_color_bgr}",
                    verbose=verbose,
                )

                roi_gray = np.zeros_like(img_gray)
                roi_indices = roi_mask == 255
                if not np.any(roi_indices):
                    log_message(f"Skipping detection {detection.get('bbox')} due to empty ROI mask.", verbose=verbose)
                    continue
                roi_gray[roi_indices] = img_gray[roi_indices]

                roi_for_thresholding = cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray

                thresholded_roi = np.zeros_like(img_gray)

                if use_otsu_threshold:
                    if np.any(roi_indices):
                        roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
                        if roi_pixels_for_otsu.size > 0:
                            thresh_val, _ = cv2.threshold(
                                roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                            )
                            log_message(f"  - Otsu threshold determined: {thresh_val}", verbose=verbose)
                            _, thresholded_roi = cv2.threshold(roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY)
                        else:
                            log_message("  - Skipping Otsu: No pixels in ROI for thresholding.", verbose=verbose)
                    else:
                        log_message("  - Skipping Otsu: ROI mask is empty.", verbose=verbose)
                else:
                    fixed_threshold = 210
                    _, thresholded_roi = cv2.threshold(roi_for_thresholding, fixed_threshold, 255, cv2.THRESH_BINARY)

                thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

                final_mask = None
                eroded_constraint_mask = cv2.erode(
                    original_yolo_mask, constraint_erosion_kernel, iterations=constraint_erosion_iterations
                )

                if is_black_bubble:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying BLACK bubble refinement logic.", verbose=verbose
                    )

                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    refined_background_shape_mask = None
                    if contours:
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                        if valid_contours:
                            largest_contour = max(valid_contours, key=cv2.contourArea)
                            refined_background_shape_mask = np.zeros_like(img_gray)
                            cv2.drawContours(
                                refined_background_shape_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
                            )
                            log_message("  - Found refined background shape.", verbose=verbose)
                        else:
                            log_message("  - No valid contours found for refined shape.", verbose=verbose)
                    else:
                        log_message("  - No contours found at all for refined shape.", verbose=verbose)

                    base_mask = original_yolo_mask.copy()
                    if refined_background_shape_mask is not None:
                        dilated_shape_mask = cv2.dilate(refined_background_shape_mask, closing_kernel, iterations=1)
                        refined_mask = cv2.bitwise_and(base_mask, dilated_shape_mask)
                        log_message("  - Refined mask using dilated shape intersection.", verbose=verbose)
                    else:
                        refined_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
                        log_message("  - No refined shape, using closed base mask (1 iter).", verbose=verbose)

                    internal_erosion_iterations = max(1, (constraint_erosion_iterations + 2))  # чуть сильнее
                    eroded_refined_mask = cv2.erode(
                        refined_mask, constraint_erosion_kernel, iterations=internal_erosion_iterations
                    )
                    closed_eroded_refined_mask = cv2.morphologyEx(
                        eroded_refined_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations
                    )
                    final_mask = cv2.bitwise_and(closed_eroded_refined_mask, eroded_constraint_mask)

                else:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying WHITE bubble refinement logic (original).",
                        verbose=verbose,
                    )
                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                        if valid_contours:
                            largest_contour = max(valid_contours, key=cv2.contourArea)
                            interior_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.drawContours(interior_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                            closed_mask = cv2.morphologyEx(
                                interior_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations
                            )
                            final_mask = cv2.bitwise_and(closed_mask, eroded_constraint_mask)
                            log_message(
                                "  - Generated mask from largest contour, closed, and constrained.", verbose=verbose
                            )
                        else:
                            log_message("  - No valid contours found.", verbose=verbose)
                    else:
                        log_message("  - No contours found at all.", verbose=verbose)

                if final_mask is not None:
                    lir_coords = None
                    if np.any(final_mask):
                        try:
                            lir_coords = largestinteriorrectangle.lir(final_mask.astype(bool))
                            log_message(f"  - Calculated LIR: {lir_coords}", verbose=verbose)
                        except Exception as lir_e:
                            log_message(f"  - LIR calculation failed: {lir_e}", verbose=verbose, is_error=True)
                            lir_coords = None
                    else:
                        log_message("  - Skipping LIR calculation: Final mask is empty.", verbose=verbose)

                    processed_bubbles.append(
                        {
                            "mask": final_mask,
                            "color": fill_color_bgr,
                            "bbox": detection.get("bbox"),
                            "lir_bbox": lir_coords,
                        }
                    )
                    log_message(
                        f"Detection {detection.get('bbox')}: Stored final mask, color {fill_color_bgr}, "
                        f"and LIR {lir_coords}.",
                        verbose=verbose,
                    )

            except Exception as e:
                error_msg = f"Error processing mask for detection {detection.get('bbox')}: {e}"
                log_message(error_msg, always_print=True, is_error=True)

        # Заливка
        for bubble_info in processed_bubbles:
            mask = bubble_info["mask"]
            color_bgr = bubble_info["color"]
            num_channels = cleaned_image.shape[2] if len(cleaned_image.shape) == 3 else 1

            if num_channels == 4 and len(color_bgr) == 3:
                fill_color = (*color_bgr, 255)
            else:
                fill_color = color_bgr

            fill_color_image = np.full_like(cleaned_image, fill_color)
            cleaned_image = np.where(np.expand_dims(mask, axis=2) == 255, fill_color_image, cleaned_image)

        return cleaned_image, processed_bubbles

    except IOError as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error cleaning speech bubbles: {str(e)}")
