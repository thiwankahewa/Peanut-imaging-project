import os
import re
import pickle
from collections import Counter

import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO


# -------------------------
# Utils
# -------------------------
def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any numeric image to uint8 using robust percentile scaling."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8) * 255.0
    return img.astype(np.uint8)


def npy_to_yolo_rgb_u8(data: np.ndarray) -> np.ndarray:
    """
    Convert npy cube (H,W,3) -> uint8 RGB image for YOLO inference.
    IMPORTANT: This should match exactly how you prepared images for YOLO training.
    Here we assume channels are [405, 720, 760] -> [R, G, B].
    """
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {data.shape}")

    img405 = to_uint8(data[:, :, 0])
    img720 = to_uint8(data[:, :, 1])
    img760 = to_uint8(data[:, :, 2])

    rgb = np.stack([img405, img720, img760], axis=2)  # RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return rgb


def channelwise_minmax_01(pixels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = pixels.min(axis=0)
    mx = pixels.max(axis=0)
    return (pixels - mn) / (mx - mn + eps)


def reg_value_to_class(v: float, class_names=None) -> str:
    if class_names is None:
        class_names = ["white", "yellow", "orange", "brown", "black"]
    centers = np.linspace(0.0, 1.0, len(class_names))
    idx = int(np.argmin(np.abs(centers - v)))
    return class_names[idx]


def draw_label_box(img_bgr: np.ndarray, bbox, text: str):
    x, y, w, h = bbox
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(img_bgr, (x, y0), (x + tw + 6, y0 + th + baseline + 4), (0, 255, 0), -1)
    cv2.putText(img_bgr, text, (x + 3, y0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)


def visualize_binary_mask(h: int, w: int, instance_masks_01, out_path: str):
    """Save a union binary mask (255=fg, 0=bg)."""
    union = np.zeros((h, w), dtype=np.uint8)
    for m in instance_masks_01:
        union = np.maximum(union, (m.astype(np.uint8) * 255))
    cv2.imwrite(out_path, union)

# -------------------------
# Functions for mask deduplication and refinement
# -------------------------
def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    m1, m2: uint8/bool masks with shape (H,W), values 0/1 or False/True
    """
    a = m1.astype(bool)
    b = m2.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)


def dedup_by_mask_iou(masks01: np.ndarray, scores: np.ndarray = None, iou_thr: float = 0.0):
    """
    masks01: (N,H,W) 0/1 uint8 or bool
    scores: (N,) higher is better; if None, keep earlier ones
    iou_thr:
      - 0.0 means: if IoU > 0 => overlap exists => remove duplicates
      - e.g. 0.7 means: only remove near-identical masks
    Returns:
      keep_indices: list[int]
    """
    N = masks01.shape[0]
    if scores is None:
        order = list(range(N))
    else:
        order = list(np.argsort(-scores))  # high -> low

    keep = []
    for i in order:
        mi = masks01[i]
        drop = False
        for j in keep:
            mj = masks01[j]
            if mask_iou(mi, mj) > iou_thr:
                drop = True
                break
        if not drop:
            keep.append(i)

    # keep indices in the original order (optional, nicer for downstream)
    keep_sorted = sorted(keep)
    return keep_sorted


def fill_holes(binary_255: np.ndarray) -> np.ndarray:
    """Fill holes inside foreground regions in a 0/255 binary image."""
    h, w = binary_255.shape
    inv = cv2.bitwise_not(binary_255)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ffmask, (0, 0), 0)
    # Remaining white in inv corresponds to holes in original
    return cv2.bitwise_or(binary_255, inv)


def refine_mask_in_bbox(seg_img_u8: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 2):
    """
    Stage-2 refinement: re-segment ONLY within bbox ROI to get a solid mask.
    Returns: (roi_mask_0_1, x0, y0, x1, y1)
    """
    H, W = seg_img_u8.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    roi = seg_img_u8[y0:y1, x0:x1]

    # Mild denoise
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # Otsu in ROI is usually very stable once bbox is correct
    _, m = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Auto invert if most of ROI becomes foreground (wrong polarity)
    if (m.sum() / 255.0) > (m.size * 0.7):
        m = cv2.bitwise_not(m)

    # Close to heal cracks and small holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Keep only largest connected component (removes grid remnants/noise)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        m = (lab == best).astype(np.uint8) * 255

    # Fill holes to make peanut solid
    m = fill_holes(m)

    roi_mask = (m > 0).astype(np.uint8)  # 0/1
    return roi_mask, x0, y0, x1, y1


def refine_yolo_instance_with_760(
    seg_img_u8: np.ndarray,
    bbox,                 # (x, y, w, h)
    yolo_mask01: np.ndarray,  # (H,W) 0/1
    pad: int = 5,
    constrain_with_yolo: bool = True,
    yolo_dilate_k: int = 7
) -> np.ndarray:
    """
    Refine one YOLO instance mask using 760nm channel within an expanded bbox ROI.
    Returns: refined full-size mask (H,W) 0/1
    """
    x, y, w, h = bbox
    roi_mask01, x0, y0, x1, y1 = refine_mask_in_bbox(seg_img_u8, x, y, w, h, pad=pad)

    H, W = seg_img_u8.shape
    refined_full = np.zeros((H, W), dtype=np.uint8)
    refined_full[y0:y1, x0:x1] = roi_mask01

    if constrain_with_yolo:
        # dilate YOLO mask slightly to avoid over-shrinking
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (yolo_dilate_k, yolo_dilate_k))
        yolo_dil = cv2.dilate((yolo_mask01 * 255).astype(np.uint8), kernel, iterations=1)
        yolo_dil01 = (yolo_dil > 0).astype(np.uint8)

        refined_full = (refined_full & yolo_dil01).astype(np.uint8)

    return refined_full

# -------------------------
# YOLO segmentation wrapper
# -------------------------
def segment_with_yolo_from_npy(
    data: np.ndarray,
    seg_img_u8: np.ndarray,
    yolo_model: YOLO,
    device=0,
    conf=0.25,
    iou_thresh=0,
    imgsz=1280,
    max_det=104,
    retina_masks=True,
    refine_with_760=True,
    refine_pad=5,
    area_min=100
):
    """
    Input: data (H,W,3) raw npy cube
    Output:
      instance_masks: list of full-size mask (H,W) in 0/1 uint8
      bboxes: list of (x, y, w, h)
      rgb_u8: the RGB image fed to YOLO (uint8)
    """
    rgb_u8 = npy_to_yolo_rgb_u8(data)
    h, w = rgb_u8.shape[:2]

    # Ultralytics can take numpy arrays directly.
    # Keep model loaded once outside this function.
    results = yolo_model.predict(
        source=rgb_u8,
        device=device,
        conf=conf,
        iou=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        retina_masks=retina_masks,
        verbose=False
    )

    r = results[0]
    instance_masks = []
    bboxes = []

    # No detections
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return instance_masks, bboxes, rgb_u8

    # masks: (N,H,W) float/bool tensor
    masks = r.masks.data  # torch tensor
    boxes = r.boxes.xyxy  # torch tensor (N,4)

    masks_np = masks.detach().cpu().numpy()
    boxes_np = boxes.detach().cpu().numpy()

    # #################################
    # -------------------------
    # YOLO masks deduplication --- These's deficiency in YOLO outputs.
    # -------------------------
    # threshold masks -> 0/1
    masks01 = (masks_np > 0.5).astype(np.uint8)   # (N,H,W)

    # get confidence scores if available
    try:
        confs = r.boxes.conf.detach().cpu().numpy()
    except Exception:
        confs = None

    # explicit mask-level NMS by IoU
    keep_idx = dedup_by_mask_iou(masks01, scores=confs, iou_thr=iou_thresh)

    # apply filtering
    masks_np = masks_np[keep_idx]
    boxes_np = boxes_np[keep_idx]
    # ##################################

    # Ensure masks match original image size (retina_masks=True helps)
    # If mismatch occurs, resize masks to (h,w)
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        m01 = (m > 0.5).astype(np.uint8)  # 0/1

        x1, y1, x2, y2 = boxes_np[i]
        x1 = int(np.clip(np.floor(x1), 0, w - 1))
        y1 = int(np.clip(np.floor(y1), 0, h - 1))
        x2 = int(np.clip(np.ceil(x2), 0, w - 1))
        y2 = int(np.clip(np.ceil(y2), 0, h - 1))

        bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        if refine_with_760:
            final_m01 = refine_yolo_instance_with_760(
                seg_img_u8=seg_img_u8,
                bbox=bbox,
                yolo_mask01=m01,
                pad=refine_pad,
                constrain_with_yolo=True, 
                yolo_dilate_k=7
            )
        else:
            final_m01 = m01

        # -------- area filtering --------
        if final_m01.sum() < area_min:
            continue

        instance_masks.append(final_m01)
        bboxes.append(bbox)

    return instance_masks, bboxes, rgb_u8


# -------------------------
# Main workflow
# -------------------------
def process_one_npy(data: np.ndarray, reg_model, yolo_model: YOLO, out_dir: str,timestamp: str,
                    normalize_pixels=True,
                    device=0, conf=0.2, iou_thresh=0, imgsz=640, max_det=104, area_min=100):
    os.makedirs(out_dir, exist_ok=True)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {data.shape}")

    img405 = data[:, :, 0]
    img720 = data[:, :, 1]
    img760 = data[:, :, 2]

    seg_img = to_uint8(img760)

    # YOLO segmentation
    instance_masks, bboxes, rgb_u8 = segment_with_yolo_from_npy(
        data=data,
        seg_img_u8=seg_img,
        yolo_model=yolo_model,
        device=device,
        conf=conf,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        retina_masks=True,
        refine_with_760=True,
        refine_pad=5,
        area_min=100
    )

    h, w = rgb_u8.shape[:2]

    # For your annotation background, keep using 760 grayscale (as before)
    annotated = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

    class_names = ["white", "yellow", "orange", "brown", "black"]
    class_counter = Counter()
    peanut_preds = []

    for mask01, bbox in zip(instance_masks, bboxes):
        mask = mask01.astype(bool)

        if mask.sum() < 10:
            continue

        pixels = np.stack([img405[mask], img720[mask], img760[mask]], axis=1).astype(np.float32)

        if normalize_pixels:
            pixels = channelwise_minmax_01(pixels)

        y_pred_pix = reg_model.predict(pixels)
        y_mean = float(np.mean(y_pred_pix))
        peanut_preds.append(y_mean)

        cls = reg_value_to_class(y_mean, class_names=class_names)
        class_counter[cls] += 1

        draw_label_box(annotated, bbox, f"{cls}")

    # Save outputs
    binary_mask_path = os.path.join(out_dir, f"{timestamp}_binary_mask.png")
    visualize_binary_mask(h, w, instance_masks, binary_mask_path)

    annotated_path = os.path.join(out_dir, f"{timestamp}_annotated.png")
    cv2.imwrite(annotated_path, annotated)

    # Summary
    n_peanuts = sum(class_counter.values())
    summary_lines = []
    summary_lines.append(f"File: {timestamp}")
    summary_lines.append(f"Peanut number: {n_peanuts}")

    if n_peanuts == 0:
        summary_lines.append("No peanuts detected.")
    else:
        for c in class_names:
            cnt = class_counter.get(c, 0)
            pct = 100.0 * cnt / n_peanuts
            summary_lines.append(f"{c}: {cnt} ({pct:.1f}%)")

        if len(peanut_preds) > 0:
            mean_maturity = float(np.mean(peanut_preds))
            std_maturity = float(np.std(peanut_preds))
            summary_lines.append(f"Mean predicted maturity (0~1): {mean_maturity:.3f}")
            summary_lines.append(f"Std predicted maturity (0~1): {std_maturity:.3f}")

            days_left = 17  # example linear mapping, formula will be determined in future
            summary_lines.append(
                f"According to the peanut maturity board, this batch of peanuts still has "
                f"approximately {days_left} days until digging."
            )

    summary_txt = "\n".join(summary_lines)
    print("\n" + "=" * 60)
    print(summary_txt)
    print("=" * 60 + "\n")

    return class_counter
