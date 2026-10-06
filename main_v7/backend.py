"""Camera, calibration, capture and analysis services for the v7 interface.

The camera settings and legacy analysis functions are carried over from v6.
Importing this module does not claim GPIO pins, open cameras, or create Tk.
Only ImagingBackend's serialized worker accesses hardware. UI updates travel
through an event queue, never through calls to Tk from a worker thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Any, TYPE_CHECKING
import importlib
import json


import os
import platform
import time
import threading
import cv2
import numpy as np
import subprocess

from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Analysis plots are rendered in the capture worker.
import matplotlib.pyplot as plt

from maturity_analysis import MaturityAnalyzer, discover_model_files
from main_v7.scan_clock import ScanClock

if TYPE_CHECKING:
    from ultralytics import YOLO

# ============================================================
#  CONFIG
# ============================================================

# Paths are rooted at the project, regardless of the launch directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_DIR / "images"
CALIB_DIR = PROJECT_DIR / "calibration"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"

MODEL_DIR = PROJECT_DIR / "models"
YOLO_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "04_26_26_peanut_seg.pt")
DEFAULT_MATURITY_MODEL = "pooled_reference_model.pkl"


# Relay pin definitions (BCM)
DRIVER_PIN = 17
LED1_PIN   = 22
LED2_PIN   = 23
LED3_PIN   = 27
LED4_PIN   = 24

# USB camera index/device
USB_CAM_INDEX = 0
USB_CAM_WIDTH = 1920
USB_CAM_HEIGHT = 1080
USB_CAM_AUTO_EXPOSURE = 3   # manual exposure mode for best preview
USB_CAM_GAMMA = 144
USB_CAM_GAIN = 0

# Hardware handles are populated only by the Connect operation.
PySpin = None
driver = led1 = led2 = led3 = led4 = None

# FLIR globals
CAM_OK = False
CAM_ERROR_MSG = ""
system = None
cam_list = None
cam = None
processor = None

# USB camera globals
USB_CAM_OK = False
USB_CAM_ERROR_MSG = ""
usb_cam = None

TRAY_ROI = (247, 125,1917, 1430)
USB_TRAY_ROI = (216, 663, 858, 1519)

LED_EXPOSURE_US = {1: 17887.0, 2: 16000.0, 3: 11000.0, 4: 10000.0}
LED_GAIN_DB = {1: 12.0, 2: 0.0, 3: 0.0, 4: 0.0}
LED_EXPOSURE_US_CAL = {1: 8000.0, 2: 13500.0, 3: 9000.0, 4: 10000.0}
LED_GAIN_DB_CAL = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

calibration_flats = {}

# YOLO / analysis parameters

ANALYSIS_DEVICE = "cpu"        # use "cpu" if CUDA is not available
ANALYSIS_CONF = 0.25
ANALYSIS_IOU_THRESH = 0
ANALYSIS_IMGSZ = 640
ANALYSIS_MAX_DET = 104
ANALYSIS_AREA_MIN = 500

analysis_runner = None


# ============================================================
#  CAMERA HELPERS
# ============================================================

def reset_camera():
    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor

    try:
        if cam is not None:
            cam.DeInit()
    except Exception:
        pass

    cam = None

    try:
        if cam_list is not None:
            cam_list.Clear()
    except Exception:
        pass

    try:
        if system is not None:
            system.ReleaseInstance()
    except Exception:
        pass

    CAM_OK = False
    CAM_ERROR_MSG = ""
    system = cam_list = cam = processor = None

def init_camera():
    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor

    reset_camera()

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        if cam_list.GetSize() == 0:
            CAM_ERROR_MSG = "No FLIR camera found"
            print("[Init] No FLIR cameras detected.")
            return

        cam = cam_list.GetByIndex(0)
        cam.Init()
        CAM_OK = True

        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        processor = PySpin.ImageProcessor()

        print("[Init] FLIR camera init OK")

    except Exception as e:
        CAM_ERROR_MSG = f"FLIR camera init error: {e!r}"
        CAM_OK = False
        print(CAM_ERROR_MSG)

def reset_usb_camera():
    global USB_CAM_OK, USB_CAM_ERROR_MSG, usb_cam

    try:
        if usb_cam is not None:
            usb_cam.release()
    except Exception:
        pass

    usb_cam = None
    USB_CAM_OK = False
    USB_CAM_ERROR_MSG = ""


def run_v4l2_controls(args):
    try:
        subprocess.run([
            "v4l2-ctl",
            "-d", f"/dev/video{USB_CAM_INDEX}"
        ] + args, check=False)
    except Exception as e:
        print(f"[USB] v4l2-ctl failed: {e}")


def init_usb_camera():
    global USB_CAM_OK, USB_CAM_ERROR_MSG, usb_cam

    print("[Init] Setting up USB camera ...")
    reset_usb_camera()

    try:
        # Force camera format before opening with OpenCV
        subprocess.run([
            "v4l2-ctl",
            "-d", f"/dev/video{USB_CAM_INDEX}",
            "--set-fmt-video=width=1920,height=1080,pixelformat=MJPG"
        ], check=False)

        time.sleep(0.2)

        usb_cam = cv2.VideoCapture(USB_CAM_INDEX, cv2.CAP_V4L2)
        if not usb_cam.isOpened():
            USB_CAM_ERROR_MSG = f"Could not open USB camera at index {USB_CAM_INDEX}"
            print("[Init]", USB_CAM_ERROR_MSG)
            USB_CAM_OK = False
            return

        run_v4l2_controls([
            "-c", f"auto_exposure={USB_CAM_AUTO_EXPOSURE}",
            "-c", f"gamma={USB_CAM_GAMMA}",
            "-c", f"gain={USB_CAM_GAIN}"
        ])

        usb_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        time.sleep(0.5)

        actual_w = int(usb_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(usb_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(usb_cam.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)])

        print(f"[USB] Actual resolution: {actual_w} x {actual_h}")
        print(f"[USB] Actual FOURCC    : {fourcc_str}")

        for _ in range(20):
            usb_cam.read()
            time.sleep(0.03)

        USB_CAM_OK = True
        print("[Init] USB camera init OK")

    except Exception as e:
        USB_CAM_ERROR_MSG = f"USB camera init error: {e!r}"
        USB_CAM_OK = False
        print(USB_CAM_ERROR_MSG)

def set_led_camera_params(led_id: int):
    if cam is None:
        return
    exp = LED_EXPOSURE_US.get(led_id, None)
    gain = LED_GAIN_DB.get(led_id, None)
    if exp is not None:
        cam.ExposureTime.SetValue(exp)
    if gain is not None:
        cam.Gain.SetValue(gain)

def set_led_camera_cal_params(led_id: int):
    if cam is None:
        return
    exp = LED_EXPOSURE_US_CAL.get(led_id, None)
    gain = LED_GAIN_DB_CAL.get(led_id, None)
    if exp is not None:
        cam.ExposureTime.SetValue(exp)
    if gain is not None:
        cam.Gain.SetValue(gain)

def capture_image():
    global CAM_OK, CAM_ERROR_MSG
    if not CAM_OK or cam is None or processor is None:
        raise RuntimeError("FLIR camera not initialized")
    image = None
    acquiring = False
    try:
        cam.BeginAcquisition()
        acquiring = True
        image = cam.GetNextImage(1000)
        if image.IsIncomplete():
            raise RuntimeError("Incomplete FLIR image; reconnect the camera and try again.")
        # Copy before releasing the Spinnaker image buffer.
        array = processor.Convert(image, PySpin.PixelFormat_Mono8).GetNDArray().copy()
        x1, y1, x2, y2 = TRAY_ROI
        return array[y1:y2, x1:x2]
    except Exception as exc:
        CAM_OK = False
        CAM_ERROR_MSG = str(exc)
        raise
    finally:
        try:
            if image is not None:
                image.Release()
        finally:
            if acquiring:
                cam.EndAcquisition()


def capture_usb_image():
    global USB_CAM_OK, USB_CAM_ERROR_MSG, usb_cam

    if not USB_CAM_OK or usb_cam is None:
        raise RuntimeError("USB camera not initialized")

    run_v4l2_controls([
        "-c", f"auto_exposure={USB_CAM_AUTO_EXPOSURE}",
        "-c", f"gamma={USB_CAM_GAMMA}",
        "-c", f"gain={USB_CAM_GAIN}"
    ])

    frame = None

    # Throw away several frames so exposure can settle
    for _ in range(20):
        ret, frame = usb_cam.read()
        if not ret:
            time.sleep(0.03)
            continue
        time.sleep(0.03)


    if not ret or frame is None:
        USB_CAM_OK = False
        USB_CAM_ERROR_MSG = "Failed to capture image from USB camera"
        raise RuntimeError(USB_CAM_ERROR_MSG)

    x1, y1, x2, y2 = USB_TRAY_ROI
    img_crop = frame[y1:y2, x1:x2]

    img_rot = cv2.rotate(img_crop, cv2.ROTATE_90_CLOCKWISE)

    return frame

def flat_field_normalize(img: np.ndarray, led_id: int):
    """
    Normalize an image using the latest calibration flat for this LED.
    N(x,y) = I(x,y) * (mean(flat) / flat(x,y))
    """
    cal = calibration_flats.get(led_id, None)
    if cal is None:
        return img.copy(), None

    img = img.astype(np.float32)
    flat = cal.astype(np.float32)

    flat_safe = np.where(flat < 1.0, 1.0, flat)
    flat_mean = flat_safe.mean()

    norm = img * (flat_mean / flat_safe)
    norm_ratio = img / flat_safe
    norm = np.clip(norm, 0, 255).astype("uint8")
    return norm, norm_ratio

def calib_flat_path_ref(led_id: int) -> str:
    return os.path.join(CALIB_DIR, f"LED{led_id}_flat_ref.npy")

def load_calibration_flats():
    global calibration_flats
    calibration_flats = {}
    for led_id in (1, 2, 3):
        path = calib_flat_path_ref(led_id)
        if os.path.exists(path):
            try:
                arr = np.load(path)
                calibration_flats[led_id] = arr
                print(f"[Calib] Loaded reference flat for LED {led_id} from {path}")
            except Exception as e:
                print(f"[Calib] Failed to load flat for LED {led_id}: {e}")

def save_calibration_flat(led_id: int, arr: np.ndarray, as_reference_if_missing=True):
    ref_path = calib_flat_path_ref(led_id)
    np.save(ref_path, arr)
    print(f"[Calib] Saved ref flat for LED {led_id} -> {ref_path}")

def cleanup_hardware():
    print("[Cleanup] Releasing hardware...")

    # GPIO
    for dev in [driver, led1, led2, led3, led4]:
        try:
            dev.off()
        except Exception:
            pass

    try:
        driver.off()
    except Exception:
        pass

    for dev in [driver, led1, led2, led3, led4]:
        try:
            dev.close()
        except Exception:
            pass

    # FLIR camera
    try:
        if CAM_OK and cam is not None:
            cam.DeInit()
    except Exception:
        pass

    try:
        if cam_list is not None:
            cam_list.Clear()
    except Exception:
        pass

    try:
        if system is not None:
            system.ReleaseInstance()
    except Exception:
        pass

    # USB camera
    try:
        if usb_cam is not None:
            usb_cam.release()
    except Exception:
        pass

    print("[Cleanup] GPIO and cameras released.")


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
    Channels are assumed to be [405, 720, 760] -> [R, G, B].
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
    """Normalize pixels channel-wise to 0-1 range."""
    mn = pixels.min(axis=0)
    mx = pixels.max(axis=0)
    return (pixels - mn) / (mx - mn + eps)


def draw_label_box(img_bgr: np.ndarray, bbox, text: str):
    """Draw bounding box and text label on the output image."""
    x, y, w, h = bbox
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(img_bgr, (x, y0), (x + tw + 6, y0 + th + baseline + 4), (0, 255, 0), -1)
    cv2.putText(img_bgr, text, (x + 3, y0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)


def visualize_binary_mask(orig_h: int, orig_w: int, instance_masks_01, out_path: str):
    """Save a union binary mask (255=foreground, 0=background) at original scale."""
    union = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for m in instance_masks_01:
        union = np.maximum(union, (m.astype(np.uint8) * 255))
    cv2.imwrite(out_path, union)


# -------------------------
# Functions for mask deduplication and refinement
# -------------------------
def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two binary masks."""
    a = m1.astype(bool)
    b = m2.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)


def dedup_by_mask_iou(masks01: np.ndarray, scores: np.ndarray = None, iou_thr: float = 0.0):
    """Remove overlapping masks based on an IoU threshold."""
    N = masks01.shape[0]
    if scores is None:
        order = list(range(N))
    else:
        order = list(np.argsort(-scores))  # Sort high to low confidence

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

    return sorted(keep)


def fill_holes(binary_255: np.ndarray) -> np.ndarray:
    """Fill holes inside foreground regions in a 0/255 binary image."""
    h, w = binary_255.shape
    inv = cv2.bitwise_not(binary_255)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ffmask, (0, 0), 0)
    return cv2.bitwise_or(binary_255, inv)


def refine_mask_in_bbox(seg_img_u8: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 2):
    """Stage-2 refinement: re-segment ONLY within bbox ROI to get a solid mask."""
    H, W = seg_img_u8.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    roi = seg_img_u8[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, m = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if (m.sum() / 255.0) > (m.size * 0.7):
        m = cv2.bitwise_not(m)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=2)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        m = (lab == best).astype(np.uint8) * 255

    m = fill_holes(m)
    roi_mask = (m > 0).astype(np.uint8)
    return roi_mask, x0, y0, x1, y1


def refine_yolo_instance_with_760(
    seg_img_u8: np.ndarray,
    bbox,
    yolo_mask01: np.ndarray,
    pad: int = 5,
    constrain_with_yolo: bool = True,
    yolo_dilate_k: int = 7
) -> np.ndarray:
    """Refine one YOLO instance mask using the 760nm channel."""
    x, y, w, h = bbox
    roi_mask01, x0, y0, x1, y1 = refine_mask_in_bbox(seg_img_u8, x, y, w, h, pad=pad)

    H, W = seg_img_u8.shape
    refined_full = np.zeros((H, W), dtype=np.uint8)
    refined_full[y0:y1, x0:x1] = roi_mask01

    if constrain_with_yolo:
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
    imgsz=640,
    max_det=104,
    retina_masks=True,
    refine_with_760=True,
    refine_pad=5,
    area_min=100
):
    """
    Run YOLO on warped 640x640 image, then map ALL outputs (masks & boxes)
    back to the exact ORIGINAL resolution before returning.
    """
    # RGB image at original dimensions
    rgb_orig = npy_to_yolo_rgb_u8(data)
    orig_h, orig_w = rgb_orig.shape[:2]

    # Explicitly warp the image to match YOLO training dimensions (ignoring aspect ratio)
    rgb_resized = cv2.resize(rgb_orig, (imgsz, imgsz))

    # Perform YOLO prediction on the 640x640 warped image
    results = yolo_model.predict(
        source=rgb_resized,
        device=device,
        conf=conf,
        iou=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        retina_masks=retina_masks,
        verbose=False
    )

    r = results[0]
    instance_masks_orig_scale = []
    bboxes_orig_scale = []

    # If no detections occur
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return instance_masks_orig_scale, bboxes_orig_scale, rgb_orig

    # Extract masks and boxes (these correspond to the 640x640 warped space)
    masks_np = r.masks.data.detach().cpu().numpy()
    boxes_np = r.boxes.xyxy.detach().cpu().numpy()

    # Threshold masks to binary 0/1
    masks01 = (masks_np > 0.5).astype(np.uint8)

    # Get confidence scores for deduplication
    try:
        confs = r.boxes.conf.detach().cpu().numpy()
    except Exception:
        confs = None

    # Mask-level NMS by IoU
    keep_idx = dedup_by_mask_iou(masks01, scores=confs, iou_thr=iou_thresh)

    # Filtered arrays
    masks_filtered = masks_np[keep_idx]
    boxes_filtered = boxes_np[keep_idx]

    # Calculate ratios to map coordinates back to the ORIGINAL dimensions
    scale_x = orig_w / imgsz
    scale_y = orig_h / imgsz

    for i in range(masks_filtered.shape[0]):
        m = masks_filtered[i]

        # MAPPING BACK MASK: Resize the mask back to original space
        m_orig_size = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        m01 = (m_orig_size > 0.5).astype(np.uint8)

        # MAPPING BACK BBOX: Scale coordinates back to original space
        x1, y1, x2, y2 = boxes_filtered[i]
        x1 = int(np.clip(np.floor(x1 * scale_x), 0, orig_w - 1))
        y1 = int(np.clip(np.floor(y1 * scale_y), 0, orig_h - 1))
        x2 = int(np.clip(np.ceil(x2 * scale_x), 0, orig_w - 1))
        y2 = int(np.clip(np.ceil(y2 * scale_y), 0, orig_h - 1))

        bbox_orig = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        # Refine the re-scaled mask against the original 760nm channel
        if refine_with_760:
            final_m01 = refine_yolo_instance_with_760(
                seg_img_u8=seg_img_u8,
                bbox=bbox_orig,
                yolo_mask01=m01,
                pad=refine_pad,
                constrain_with_yolo=True,
                yolo_dilate_k=7
            )
        else:
            final_m01 = m01

        # Skip peanuts that are too small
        if final_m01.sum() < area_min:
            continue

        instance_masks_orig_scale.append(final_m01)
        bboxes_orig_scale.append(bbox_orig)

    # Return lists holding data entirely mapped back to original resolution
    return instance_masks_orig_scale, bboxes_orig_scale, rgb_orig


# -------------------------
# Main workflow
# -------------------------
def process_one_cube(data: np.ndarray, stem: str, pca_model, reg_model, yolo_model: YOLO, out_dir: str,
                     normalize_pixels=True,
                     device=0, conf=0.2, iou_thresh=0, imgsz=640, max_det=104, area_min=100):

    os.makedirs(out_dir, exist_ok=True)

    fname = f"{stem}.npy"

    # 1. Use captured 3-channel cube
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {data.shape} for {fname}")

    img405 = data[:, :, 0]
    img720 = data[:, :, 1]
    img760 = data[:, :, 2]

    # Original dimension 760nm image for background / mask refinement
    seg_img_orig_scale = to_uint8(img760)

    # 2. YOLO Segmentation (internally resizes to 640x640, predicts, then maps EVERYTHING back to original scale)
    instance_masks_orig, bboxes_orig, rgb_orig = segment_with_yolo_from_npy(
        data=data,
        seg_img_u8=seg_img_orig_scale,
        yolo_model=yolo_model,
        device=device,
        conf=conf,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        retina_masks=True,
        refine_with_760=True,
        refine_pad=5,
        area_min=area_min
    )

    orig_h, orig_w = rgb_orig.shape[:2]

    # Base image for final visualizations (guaranteed to be original resolution)
    annotated = cv2.cvtColor(seg_img_orig_scale, cv2.COLOR_GRAY2BGR)

    peanut_preds = []

    # Array to store the continuous maturity map at original resolution
    maturity_map_orig = np.full((orig_h, orig_w), np.nan)

    for mask01, bbox in zip(instance_masks_orig, bboxes_orig):
        # 5-pixel mask erosion to drop noisy edge pixels
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask01.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Fallback to the original mask if erosion erases the peanut completely
        if mask_eroded.sum() < 10:
            mask_eroded = mask01.astype(bool)

        if mask_eroded.sum() < 10:
            continue

        # Extract pixels directly from the original un-resized spectral bands
        pixels = np.stack([img405[mask_eroded], img720[mask_eroded], img760[mask_eroded]], axis=1).astype(np.float32)

        if normalize_pixels:
            pixels = channelwise_minmax_01(pixels)

        # 3. PCA Projection & Regression
        pc_scores = pca_model.transform(pixels)
        pc12_scores = pc_scores[:, :2]

        y_pred_pix = reg_model.predict(pc12_scores)
        y_pred_pix = np.clip(y_pred_pix, 0, 1)

        # Populate the original-scale continuous maturity map
        maturity_map_orig[mask_eroded] = y_pred_pix

        # Calculate average maturity for the whole peanut instance
        y_mean = float(np.mean(y_pred_pix))
        peanut_preds.append(y_mean)

        # Draw box and the CONTINUOUS MATURITY INDEX label on original-scale image
        draw_label_box(annotated, bbox, f"{y_mean:.2f}")

    # 4. Save Outputs (All in original dimensions)
    binary_mask_path = os.path.join(out_dir, f"{stem}_binary_mask.png")
    visualize_binary_mask(orig_h, orig_w, instance_masks_orig, binary_mask_path)

    annotated_path = os.path.join(out_dir, f"{stem}_annotated.png")
    cv2.imwrite(annotated_path, annotated)

    # Save Continuous Maturity Map Heatmap
    plt.figure(figsize=(8,6))
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='lightgray')

    plt.imshow(maturity_map_orig, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label="Maturity Index (0-1)")
    plt.title("Peanut Continuous Maturity Map")
    plt.axis('off')

    heatmap_path = os.path.join(out_dir, f"{stem}_maturity_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Terminal summary logic using continuous indexes directly
    n_peanuts = len(peanut_preds)
    summary_lines = []
    summary_lines.append(f"File: {fname}")
    summary_lines.append(f"Original Resolution: {orig_w}x{orig_h}")
    summary_lines.append(f"Peanut number: {n_peanuts}")

    if n_peanuts == 0:
        summary_lines.append("No peanuts detected.")
    else:
        mean_maturity = float(np.mean(peanut_preds))
        std_maturity = float(np.std(peanut_preds))
        summary_lines.append(f"Mean predicted maturity index (0~1): {mean_maturity:.3f}")
        summary_lines.append(f"Std predicted maturity index (0~1): {std_maturity:.3f}")

        summary_lines.append("DUD unavailable: this regression model has no DUD calibration.")

    summary_txt = "\n".join(summary_lines)
    print("\n" + "=" * 60)
    print(summary_txt)
    print("=" * 60 + "\n")

    result = {
        "file": fname,
        "n_peanuts": n_peanuts,
        "peanut_preds": peanut_preds,
        "mean_maturity": float(np.mean(peanut_preds)) if n_peanuts else None,
        "std_maturity": float(np.std(peanut_preds)) if n_peanuts else None,
        "days_left": None,
        "binary_mask_path": binary_mask_path,
        "annotated_path": annotated_path,
        "heatmap_path": heatmap_path,
    }
    return result


def process_one_npy(npy_path: str, pca_model, reg_model, yolo_model: YOLO, out_dir: str,
                    normalize_pixels=True,
                    device=0, conf=0.2, iou_thresh=0, imgsz=640, max_det=104, area_min=100):
    """Compatibility wrapper for processing an existing .npy cube from disk."""
    data = np.load(npy_path)
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    return process_one_cube(
        data=data,
        stem=stem,
        pca_model=pca_model,
        reg_model=reg_model,
        yolo_model=yolo_model,
        out_dir=out_dir,
        normalize_pixels=normalize_pixels,
        device=device,
        conf=conf,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        max_det=max_det,
        area_min=area_min,
    )


@dataclass(frozen=True)
class CaptureRequest:
    """A snapshot of GUI selections, taken before starting the worker."""

    model_path: Path | None
    suffix: str = ""


@dataclass(frozen=True)
class BackendEvent:
    kind: str
    data: Any = None


class OperationCancelled(Exception):
    pass


OUTPUT_NAMES = {
    "digital_mpb_profile.png": "Maturity profile",
    "continuous_dud_map.png": "Days until digging",
    "continuous_maturity_index_blue_red_map.png": "Maturity · blue / red",
    "continuous_maturity_index_map.png": "Maturity · board colors",
    "eroded_masks.png": "Peanut masks",
    "annotated_pods.png": "Detected peanuts",
}

ANALYSIS_DISPLAY_NAMES = (
    "digital_mpb_profile.png",
    "continuous_maturity_index_blue_red_map.png",
)


@dataclass(frozen=True)
class SavedScan:
    name: str
    images: tuple[Path, ...]


def list_saved_scans(analysis=True):
    """Return one page per scan, newest first; analysis requires all six outputs."""
    root = Path(ANALYSIS_OUTPUT_DIR if analysis else IMAGE_DIR)
    scans = []
    if analysis:
        by_directory = {}
        for path in root.rglob("*.png"):
            if path.is_file() and path.name in OUTPUT_NAMES:
                by_directory.setdefault(path.parent, {})[path.name] = path
        for directory, images in by_directory.items():
            if OUTPUT_NAMES.keys() <= images.keys():
                scans.append(SavedScan(directory.name, tuple(images[name] for name in ANALYSIS_DISPLAY_NAMES)))
    else:
        by_capture = {}
        for path in root.glob("*.png"):
            if path.is_file():
                # Both v6 and v7 prefix every image with the capture timestamp.
                # v7 includes microseconds; data-collection labels follow LED IDs.
                capture = path.stem.split("_", 1)[0]
                by_capture.setdefault(capture, []).append(path)
        for capture, images in by_capture.items():
            scans.append(SavedScan(capture, tuple(sorted(images))))
    # Timestamp prefixes sort chronologically, regardless of copied-file mtimes.
    return sorted(scans, key=lambda scan: scan.name, reverse=True)


def saved_image_title(path, analysis=True):
    if analysis:
        return OUTPUT_NAMES.get(path.name, "Maturity heatmap")
    return path.stem.split("_", 1)[-1].replace("_", " ")


def read_saved_summary(path):
    summary_path = Path(path).parent / "results.json"
    if not summary_path.is_file():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The saved summary must be a JSON object.")
    return data


class _DesktopRelay:
    """Windows development retains v6's no-op GPIO behavior."""

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


class ImagingBackend:
    """One backend per process; all hardware operations are serialized."""

    def __init__(self):
        self.scan_clock = ScanClock()
        self.events = Queue()
        self.busy = False
        self.active_led = None
        self._closing = False
        self._power_action = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self.worker = None

    def emit(self, kind, data=None):
        self.events.put(BackendEvent(kind, data))

    @staticmethod
    def models():
        return discover_model_files(MODEL_DIR)

    def state(self):
        calibrated = self._valid_flats(calibration_flats)
        flir = CAM_OK
        usb = USB_CAM_OK
        return dict(flir=flir, usb=usb, calibrated=calibrated, busy=self.busy,
                    active_led=self.active_led,
                    ready=flir and usb and calibrated and self.active_led is None)

    @staticmethod
    def _valid_flats(flats):
        x1, y1, x2, y2 = TRAY_ROI
        expected = (y2 - y1, x2 - x1)
        return all(
            isinstance(flats.get(i), np.ndarray)
            and flats[i].shape == expected
            and np.issubdtype(flats[i].dtype, np.number)
            and np.isfinite(flats[i]).all()
            and float(flats[i].mean()) > 0
            for i in (1, 2, 3)
        )

    def connect(self):
        return self._submit("connect", self._connect)

    def calibrate(self):
        return self._submit("calibrate", self._calibrate)

    def capture(self, request: CaptureRequest):
        return self._submit("capture", lambda: self._capture(request))

    def toggle_led(self, led_id: int):
        if led_id not in (1, 2, 3, 4):
            raise ValueError("Choose LED 1, 2, 3, or 4.")
        return self._submit("led", lambda: self._toggle_led(led_id))

    def _submit(self, operation, action):
        with self._lock:
            if self.busy or self._closing:
                return False
            self.busy = True
        self.emit("state", self.state())
        self.worker = threading.Thread(target=self._run, args=(operation, action), daemon=False)
        self.worker.start()
        return True

    def _run(self, operation, action):
        failed = False
        try:
            action()
        except OperationCancelled:
            failed = True
            self.emit("status", "Closing after the current step…")
        except Exception as exc:
            failed = True
            self.emit("status", f"{operation.capitalize()} needs attention")
            if operation == "capture":
                self.emit("result", None)
            self.emit("error", {"title": f"{operation.capitalize()} failed", "message": str(exc)})
        finally:
            if operation != "led" or failed:
                self._lights_off()
            with self._lock:
                closing = self._closing
                if not closing:
                    self.busy = False
            if closing:
                self._shutdown()
            else:
                self.emit("state", self.state())
                self.emit("done", operation)

    def _check_cancel(self):
        if self._cancel.is_set():
            raise OperationCancelled()

    def _pause(self, seconds):
        if self._cancel.wait(seconds):
            raise OperationCancelled()

    def _lights_off(self):
        for device in (led1, led2, led3, led4, driver):
            if device is not None:
                try:
                    device.off()
                except Exception:
                    pass
        self.active_led = None

    def _ensure_hardware(self):
        global PySpin, driver, led1, led2, led3, led4
        if PySpin is None:
            try:
                PySpin = importlib.import_module("PySpin")
            except ImportError as exc:
                raise RuntimeError("The FLIR driver is unavailable. Run v7 in the existing camera environment.") from exc
        if driver is None:
            if platform.system() == "Linux":
                os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")
                from gpiozero import OutputDevice
                factory = lambda pin: OutputDevice(pin, active_high=True, initial_value=False)
            else:
                factory = lambda pin: _DesktopRelay()
            devices = []
            try:
                for pin in (DRIVER_PIN, LED1_PIN, LED2_PIN, LED3_PIN, LED4_PIN):
                    devices.append(factory(pin))
            except Exception:
                for device in devices:
                    device.close()
                raise
            driver, led1, led2, led3, led4 = devices

    def _connect(self):
        self.emit("status", "Connecting cameras…")
        self._lights_off()
        for path in (IMAGE_DIR, CALIB_DIR, ANALYSIS_OUTPUT_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)
        self._ensure_hardware()
        self._check_cancel()
        init_camera()
        self._check_cancel()
        init_usb_camera()
        load_calibration_flats()
        errors = []
        if not CAM_OK:
            errors.append(f"FLIR: {CAM_ERROR_MSG}")
        if not USB_CAM_OK:
            errors.append(f"USB: {USB_CAM_ERROR_MSG}")
        if errors:
            raise RuntimeError("\n".join(errors) + "\nCheck camera connections, then reconnect in Settings.")
        self.emit("status", "Ready to scan" if self.state()["calibrated"] else "Calibrate the camera in Settings before scanning.")

    def _calibrate(self):
        global calibration_flats
        if not self.state()["flir"]:
            raise RuntimeError("Connect the FLIR camera before calibration.")
        self._lights_off()
        flats = {}
        calibration_flats = {}
        for led_id, device in enumerate((led1, led2, led3), start=1):
            self._check_cancel()
            self.emit("status", f"Calibrating light {led_id} of 3…")
            set_led_camera_cal_params(led_id)
            try:
                driver.on()
                device.on()
                self._pause(0.3)
                image = capture_image()
            finally:
                self._lights_off()
            self._pause(0.2)
            if image is None:
                raise RuntimeError(f"Calibration image for light {led_id} is missing. Please retry calibration.")
            flats[led_id] = image
            self.emit("progress", led_id / 3)
        if not self._valid_flats(flats):
            raise RuntimeError("Calibration images are incomplete, empty, or the wrong size. Check the white board and retry.")
        self._check_cancel()
        for led_id, image in flats.items():
            save_calibration_flat(led_id, image)
            self._save_png(Path(CALIB_DIR) / f"LED{led_id}_CALIB_RAW.png", image)
        calibration_flats = flats
        self.emit("status", "Remove the calibration board")

    @staticmethod
    def _save_png(path, image):
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not save image: {path}")

    def _capture(self, request):
        global analysis_runner
        if not self.state()["ready"]:
            raise RuntimeError("Connect both cameras, calibrate all three lights, and turn off manual LED testing before scanning.")
        if request.suffix and any(character in request.suffix for character in ("/", "\\", "\x00")):
            raise ValueError("Data collection labels cannot contain path separators.")
        self.emit("result", None)
        self.emit("progress", 0)
        timestamp = self.scan_clock.now().strftime("%Y%m%d-%H%M%S-%f")
        bands, ratios = {}, {}
        for index, device in enumerate((led1, led2, led3), start=1):
            self._check_cancel()
            self.emit("status", f"Capturing light {index} of 4…")
            set_led_camera_params(index)
            try:
                driver.on()
                device.on()
                self._pause(0.3)
                image = capture_image()
            finally:
                self._lights_off()
            self._pause(0.2)
            if image is None:
                raise RuntimeError(f"Light {index} did not produce an image. Please try again.")
            normalized, ratio = flat_field_normalize(image, index)
            if ratio is None:
                raise RuntimeError(f"Missing calibration for light {index}; recalibrate in Settings.")
            bands[index], ratios[index] = normalized, ratio.astype(np.float32)
            prefix = Path(IMAGE_DIR) / f"{timestamp}_LED{index}"
            self._save_png(f"{prefix}_raw{request.suffix}.png", image)
            self._save_png(f"{prefix}_norm{request.suffix}.png", normalized)
            np.save(f"{prefix}_ratio{request.suffix}.npy", ratios[index])
            self.emit("progress", index * 0.15)
        cube_path = Path(IMAGE_DIR) / f"{timestamp}_LED123{request.suffix}.npy"
        # Preserve training representation and wavelength order: 405, 720, 760.
        np.save(cube_path, np.dstack([ratios[1], ratios[2], ratios[3]]))
        self._save_png(Path(IMAGE_DIR) / f"{timestamp}_LED123_pseudo{request.suffix}.png",
                       np.dstack([bands[3], bands[2], bands[1]]))
        self._check_cancel()
        self.emit("status", "Analyzing peanut maturity…")
        analysis_error = None
        try:
            if request.model_path is None:
                raise ValueError("No maturity model selected. Add a .pkl file in models and select it in Developer mode.")
            if analysis_runner is None:
                analysis_runner = MaturityAnalyzer(YOLO_SEG_MODEL_PATH, device=ANALYSIS_DEVICE)
            result = analysis_runner.analyze(
                npy_path=cube_path, model_path=request.model_path,
                output_dir=ANALYSIS_OUTPUT_DIR, legacy_processor=process_one_npy,
                legacy_options=dict(normalize_pixels=True, device=ANALYSIS_DEVICE,
                                    conf=ANALYSIS_CONF, iou_thresh=ANALYSIS_IOU_THRESH,
                                    imgsz=ANALYSIS_IMGSZ, max_det=ANALYSIS_MAX_DET,
                                    area_min=ANALYSIS_AREA_MIN),
            )
            self.emit("result", result)
        except Exception as exc:
            analysis_error = str(exc)
        self.emit("progress", 0.85)
        self._check_cancel()
        self.emit("status", "Capturing light 4 reference image…")
        try:
            driver.on()
            led4.on()
            self._pause(1)
            reference = capture_usb_image()
        finally:
            self._lights_off()
        self._save_png(Path(IMAGE_DIR) / f"{timestamp}_LED4{request.suffix}.png", reference)
        self.emit("progress", 1)
        self.emit("status", "Images saved; analysis needs attention" if analysis_error else "Scan complete · results saved")
        if analysis_error:
            self.emit("error", {"title": "Analysis unavailable", "message": f"Your images were saved.\n\n{analysis_error}"})

    def _toggle_led(self, led_id):
        was_active = self.active_led == led_id
        self._lights_off()
        if not was_active:
            self._ensure_hardware()
            driver.on()
            (led1, led2, led3, led4)[led_id - 1].on()
            self.active_led = led_id
        self.emit("status", f"Light {led_id} is on · tap again to turn off" if self.active_led else "All lights off")

    def request_close(self, power_action=None):
        """Cancel between hardware steps; wait for inference before releasing resources."""
        if power_action not in (None, "poweroff", "reboot"):
            raise ValueError("Unsupported power action.")
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._power_action = power_action
            self._cancel.set()
            if self.busy:
                return
            self.busy = True
        self.worker = threading.Thread(target=self._shutdown, daemon=False)
        self.worker.start()

    def _shutdown(self):
        global driver, led1, led2, led3, led4
        try:
            self._lights_off()
            reset_camera()
            reset_usb_camera()
            for device in (driver, led1, led2, led3, led4):
                if device is not None:
                    try:
                        device.close()
                    except Exception:
                        pass
            driver = led1 = led2 = led3 = led4 = None
            if self._power_action:
                self._request_system_power(self._power_action)
        except Exception as exc:
            # Keep the app open if systemd denies or cannot perform the request.
            with self._lock:
                self._closing = False
                self._power_action = None
                self._cancel.clear()
                self.busy = False
            self.emit("close_failed", {"title": "Could not close the system", "message": str(exc)})
            self.emit("state", self.state())
        else:
            self.busy = False
            self.emit("closed")

    @staticmethod
    def _request_system_power(action):
        if action not in ("poweroff", "reboot"):
            raise ValueError("Unsupported power action.")
        if platform.system() != "Linux":
            raise RuntimeError("System power controls require Linux on the imaging box.")
        try:
            subprocess.run(["systemctl", "--no-ask-password", "--no-block", action],
                           check=True, capture_output=True, text=True, timeout=15)
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "Permission denied or system unavailable.").strip()
            raise RuntimeError(f"The system did not accept the power request.\n{details}\n\n"
                               "The desktop account must be allowed to shut down or restart the Pi.") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("The system power service (systemctl) is unavailable.") from exc
