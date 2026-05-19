#!/usr/bin/env python3

#integration with shaoqi code

import os
import platform
import time
import threading
import cv2
import PySpin
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import subprocess

import pickle
import re
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

if platform.system() == "Linux":
    from gpiozero import OutputDevice
else:
    # Fake OutputDevice for Windows
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            print("[MOCK] OutputDevice created (Windows)")
        def on(self):  print("[MOCK] ON")
        def off(self): print("[MOCK] OFF")
        def close(self): print("[MOCK] CLOSE")


# ============================================================
#  CONFIG
# ============================================================

# Directory to save images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
CALIB_DIR = "calibration"
os.makedirs(CALIB_DIR, exist_ok=True)
ANALYSIS_OUTPUT_DIR = "analysis_outputs"
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

MODEL_DIR = "models"
YOLO_SEG_MODEL_PATH = os.path.join(MODEL_DIR, "04_26_26_peanut_seg.pt")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "pca12_regression_model.pkl")

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"   # Force a working GPIO backend

# Relay pin definitions (BCM)
DRIVER_PIN = 17
LED1_PIN   = 23
LED2_PIN   = 22
LED3_PIN   = 27
LED4_PIN   = 24

# USB camera index/device
USB_CAM_INDEX = 0
USB_CAM_WIDTH = 1920
USB_CAM_HEIGHT = 1200

# --- Initialize relays ---
driver = OutputDevice(DRIVER_PIN, active_high=True, initial_value=False)
led1   = OutputDevice(LED1_PIN,   active_high=True, initial_value=False)
led2   = OutputDevice(LED2_PIN,   active_high=True, initial_value=False)
led3   = OutputDevice(LED3_PIN,   active_high=True, initial_value=False)
led4   = OutputDevice(LED4_PIN,   active_high=True, initial_value=False)

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

TRAY_ROI = (274, 118, 1941, 1430)
USB_TRAY_ROI = (633, 57, 1322, 950)

LED_EXPOSURE_US = {1: 17887.0, 2: 17887.0, 3: 17887.0}
LED_GAIN_DB = {1: 12.0, 2: 9.0, 3: 6.5}
LED_EXPOSURE_US_CAL = {1: 13500.0, 2: 17887.0, 3: 17887.0}
LED_GAIN_DB_CAL = {1: 0.0, 2: 4.5, 3: 1.2}

calibration_flats = {}

analysis_models_loaded = False
pca_model = None
reg_model = None
yolo_model = None

# YOLO / analysis parameters

ANALYSIS_DEVICE = "cpu"        # use "cpu" if CUDA is not available
ANALYSIS_CONF = 0.25
ANALYSIS_IOU_THRESH = 0
ANALYSIS_IMGSZ = 640
ANALYSIS_MAX_DET = 104
ANALYSIS_AREA_MIN = 100


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

def init_usb_camera():
    global USB_CAM_OK, USB_CAM_ERROR_MSG, usb_cam

    print("[Init] Setting up USB camera ...")
    reset_usb_camera()

    try:
        # Force camera format before opening with OpenCV
        subprocess.run([
            "v4l2-ctl",
            "-d", f"/dev/video{USB_CAM_INDEX}",
            "--set-fmt-video=width=1920,height=1200,pixelformat=MJPG"
        ], check=False)

        time.sleep(0.2)

        usb_cam = cv2.VideoCapture(USB_CAM_INDEX, cv2.CAP_V4L2)
        if not usb_cam.isOpened():
            USB_CAM_ERROR_MSG = f"Could not open USB camera at index {USB_CAM_INDEX}"
            print("[Init]", USB_CAM_ERROR_MSG)
            USB_CAM_OK = False
            return

        usb_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

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
    x1, y1, x2, y2 = TRAY_ROI

    if not CAM_OK or cam is None or processor is None:
        raise RuntimeError("FLIR camera not initialized")

    try:
        cam.BeginAcquisition()
        img = cam.GetNextImage(1000)
    except Exception:
        CAM_OK = False
        CAM_ERROR_MSG = "Acquisition error: Check the FLIR camera connection"
        print("[Camera] Begin/GetNextImage failed:", CAM_ERROR_MSG)
        raise RuntimeError(CAM_ERROR_MSG)

    if img.IsIncomplete():
        print("[x] Incomplete image.")
        img.Release()
        cam.EndAcquisition()
        return None

    arr = processor.Convert(img, PySpin.PixelFormat_Mono8).GetNDArray()
    img.Release()
    cam.EndAcquisition()
    return arr[y1:y2, x1:x2]

def capture_usb_image():
    global USB_CAM_OK, USB_CAM_ERROR_MSG, usb_cam

    if not USB_CAM_OK or usb_cam is None:
        raise RuntimeError("USB camera not initialized")

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

    return img_rot

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

        days_left = 17 
        summary_lines.append(
            f"According to the peanut maturity board, this batch of peanuts still has "
            f"approximately {days_left} days until digging."
        )

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
        "days_left": 17 if n_peanuts else None,
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



# -------------------------
# Analysis model loading
# -------------------------
def load_analysis_models():
    """Load YOLO segmentation model and PCA-regression model once."""
    global analysis_models_loaded, pca_model, reg_model, yolo_model

    if analysis_models_loaded:
        return

    if not os.path.exists(YOLO_SEG_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_SEG_MODEL_PATH}. "
            "Update YOLO_SEG_MODEL_PATH or place the model file in the models folder."
        )

    if not os.path.exists(REG_MODEL_PATH):
        raise FileNotFoundError(
            f"PCA-regression model not found: {REG_MODEL_PATH}. "
            "Update REG_MODEL_PATH or place the .pkl file in the models folder."
        )

    with open(REG_MODEL_PATH, "rb") as f:
        maturity_model_dict = pickle.load(f)

    pca_model = maturity_model_dict["pca"]
    reg_model = maturity_model_dict["reg"]
    yolo_model = YOLO(YOLO_SEG_MODEL_PATH)

    analysis_models_loaded = True


# ============================================================
#  GUI APP
# ============================================================

class PeanutApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Peanut Imaging Box")

        if "windows" in platform.system().lower():
            self.geometry("800x480")
            self.resizable(False, False)
        else:
            self.attributes("-fullscreen", True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.capture_thread = None
        self.is_capturing = False
        self.preview_img = None
        self.led1_on = False
        self.led2_on = False
        self.led3_on = False
        self.led4_on = False

        # Tk variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Idle")

        self._create_style()
        self._create_widgets()

        self.start_btn.config(state="disabled")
        self.set_status("Initializing cameras…")
        self.after(300, self.startup_camera_init)

    # ---------------- Styles ----------------
    def _create_style(self):
        style = ttk.Style(self)
        style.configure("TButton", font=("Helvetica", 14))
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Start.TButton", font=("Helvetica", 36, "bold"),
                        borderwidth=0, focuscolor="", padding=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[10, 5])
        style.configure("LedOff.TButton", font=("Helvetica", 12), padding=5)
        style.configure("LedOn.TButton", font=("Helvetica", 12, "bold"),
                        padding=5, background="#4caf50", foreground="white")
        style.map("LedOn.TButton", background=[("active", "#66bb6a")])

    # ---------------- Layout ----------------
    def _create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # ==== Capture tab ====
        self.tab_capture = ttk.Frame(notebook)
        notebook.add(self.tab_capture, text="Capture")

        self.notebook = notebook

        self.tab_capture.columnconfigure(0, weight=1)
        self.tab_capture.columnconfigure(1, weight=2)
        self.tab_capture.rowconfigure(0, weight=1)

        left_cap = ttk.Frame(self.tab_capture)
        left_cap.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        left_cap.columnconfigure(0, weight=1)
        left_cap.rowconfigure(0, weight=1)
        left_cap.rowconfigure(1, weight=1)

        header_lbl = ttk.Label(
            left_cap,
            text="Peanut Imaging Software",
            style="Header.TLabel"
        )
        header_lbl.grid(row=0, column=0, pady=(0, 10), sticky="n")

        self.start_btn = ttk.Button(
            left_cap,
            text="START",
            style="Start.TButton",
            command=self.on_start_capture
        )
        self.start_btn.grid(row=1, column=0, pady=10, ipadx=10, ipady=10, sticky="n")

        self.progress_bar = ttk.Progressbar(
            self.tab_capture,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2,
                               padx=40, pady=(0, 5), sticky="ew")

        self.status_label = ttk.Label(
            self.tab_capture,
            textvariable=self.status_var
        )
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        right_cap = ttk.LabelFrame(self.tab_capture, text="Latest Results")
        right_cap.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        right_cap.columnconfigure(0, weight=1)
        for r in range(6):
            right_cap.rowconfigure(r, weight=1)

        self.total_var = tk.StringVar(value="Total peanuts: -")
        self.black_var = tk.StringVar(value="Mean maturity index: -")
        self.brown_var = tk.StringVar(value="Std maturity index: -")
        self.yellow_var = tk.StringVar(value="Estimated days to digging: -")
        self.white_var = tk.StringVar(value="Output: -")

        ttk.Label(right_cap, textvariable=self.total_var).grid(row=0, column=0, sticky="w", padx=10, pady=2)
        ttk.Label(right_cap, textvariable=self.black_var).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ttk.Label(right_cap, textvariable=self.brown_var).grid(row=2, column=0, sticky="w", padx=10, pady=2)
        ttk.Label(right_cap, textvariable=self.yellow_var).grid(row=3, column=0, sticky="w", padx=10, pady=2)
        ttk.Label(right_cap, textvariable=self.white_var).grid(row=4, column=0, sticky="w", padx=10, pady=2)

        # ==== Gallery tab ====
        self.tab_gallery = ttk.Frame(notebook)
        notebook.add(self.tab_gallery, text="Gallery")

        self.tab_gallery.columnconfigure(0, weight=1)
        self.tab_gallery.columnconfigure(1, weight=1)
        self.tab_gallery.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(self.tab_gallery)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        list_header = ttk.Label(left_frame, text="Captured Images", style="Header.TLabel")
        list_header.grid(row=0, column=0, pady=(0, 5))

        self.image_listbox = tk.Listbox(left_frame, width=30, font=("Helvetica", 11))
        self.image_listbox.grid(row=1, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        self.refresh_btn = ttk.Button(left_frame, text="Refresh List", command=self.load_image_list)
        self.refresh_btn.grid(row=2, column=0, pady=(5, 0), sticky="ew")

        right_frame = ttk.Frame(self.tab_gallery)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        center_frame = ttk.Frame(right_frame)
        center_frame.grid(row=0, column=0)

        self.preview_label = ttk.Label(center_frame, text="No image selected")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.load_image_list()

        # ==== Settings tab ====
        self.tab_settings = ttk.Frame(notebook)
        notebook.add(self.tab_settings, text="Settings")

        self.tab_settings.columnconfigure(0, weight=1)
        for r in range(4):
            self.tab_settings.rowconfigure(r, weight=1)

        leds_frame = ttk.LabelFrame(self.tab_settings, text="Manual LED Test")
        leds_frame.grid(row=1, column=0, pady=10, padx=40, sticky="ew")
        for c in range(4):
            leds_frame.columnconfigure(c, weight=1)

        self.led1_btn = tk.Button(
            leds_frame, text="LED 1", font=("Helvetica", 12), relief="raised", bd=2,
            command=lambda: self.toggle_led_exclusive(led1, "led1_on", self.led1_btn)
        )
        self.led1_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.led2_btn = tk.Button(
            leds_frame, text="LED 2", font=("Helvetica", 12), relief="raised", bd=2,
            command=lambda: self.toggle_led_exclusive(led2, "led2_on", self.led2_btn)
        )
        self.led2_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.led3_btn = tk.Button(
            leds_frame, text="LED 3", font=("Helvetica", 12), relief="raised", bd=2,
            command=lambda: self.toggle_led_exclusive(led3, "led3_on", self.led3_btn)
        )
        self.led3_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.led4_btn = tk.Button(
            leds_frame, text="LED 4", font=("Helvetica", 12), relief="raised", bd=2,
            command=lambda: self.toggle_led_exclusive(led4, "led4_on", self.led4_btn)
        )
        self.led4_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.led_default_bg = self.led1_btn.cget("bg")
        self.led_default_fg = self.led1_btn.cget("fg")

        cam_frame = ttk.LabelFrame(self.tab_settings, text="Camera")
        cam_frame.grid(row=2, column=0, pady=10, padx=40, sticky="ew")
        cam_frame.columnconfigure(0, weight=1)
        cam_frame.columnconfigure(1, weight=1)

        reconnect_btn = ttk.Button(
            cam_frame, text="Reconnect Cameras",
            command=self.on_reconnect_camera
        )
        reconnect_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        calibrate_btn = ttk.Button(
            cam_frame, text="Calibrate Camera",
            command=self.calibrate_camera
        )
        calibrate_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        exit_btn = ttk.Button(
            self.tab_settings,
            text="Exit to Desktop",
            command=self.on_close
        )
        exit_btn.grid(row=3, column=0, pady=(10, 30), ipadx=20, ipady=5)

        self.status_label = ttk.Label(
            self.tab_settings,
            textvariable=self.status_var
        )
        self.status_label.grid(row=4, column=0, columnspan=2, pady=(0, 10))

    # =======================================================
    #  Status / Progress helpers
    # =======================================================

    def set_status(self, text: str):
        self.status_var.set(text)

    def set_progress(self, value: float):
        self.progress_var.set(value)

    def safe_status(self, text: str):
        self.after(0, lambda: self.set_status(text))

    def safe_progress(self, value: float):
        self.after(0, lambda: self.set_progress(value))

    def safe_update_results(self, result: dict):
        def _update():
            n = result.get("n_peanuts", 0)
            mean = result.get("mean_maturity")
            std = result.get("std_maturity")
            days = result.get("days_left")

            self.total_var.set(f"Total peanuts: {n}")
            self.black_var.set(f"Mean maturity index: {mean:.3f}" if mean is not None else "Mean maturity index: -")
            self.brown_var.set(f"Std maturity index: {std:.3f}" if std is not None else "Std maturity index: -")
            self.yellow_var.set(f"Estimated days to digging: {days}" if days is not None else "Estimated days to digging: -")
            self.white_var.set(f"Output: {os.path.basename(result.get('annotated_path', '-'))}")
        self.after(0, _update)

    def safe_capture_end(self):
        def _end():
            self.is_capturing = False
            self.start_btn.config(state="normal")
        self.after(0, _end)

    def safe_refresh_gallery(self):
        self.after(0, self.load_image_list)

    def set_tabs_for_led_test(self, led_test_active: bool):
        if not hasattr(self, "notebook"):
            return

        try:
            if led_test_active:
                self.notebook.tab(self.tab_capture, state="disabled")
                self.notebook.tab(self.tab_gallery, state="disabled")
                self.notebook.tab(self.tab_settings, state="normal")
            else:
                self.notebook.tab(self.tab_capture, state="normal")
                self.notebook.tab(self.tab_gallery, state="normal")
                self.notebook.tab(self.tab_settings, state="normal")
        except Exception as e:
            print("set_tabs_for_led_test error:", e)

    def turn_off_all_leds(self):
        led1.off()
        led2.off()
        led3.off()
        led4.off()
        driver.off()

        self.led1_on = self.led2_on = self.led3_on = self.led4_on = False

        if hasattr(self, "led1_btn"):
            self.led1_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 1")
        if hasattr(self, "led2_btn"):
            self.led2_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 2")
        if hasattr(self, "led3_btn"):
            self.led3_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 3")
        if hasattr(self, "led4_btn"):
            self.led4_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 4")

        self.set_tabs_for_led_test(False)

    def toggle_led_exclusive(self, target_led, state_attr_name, target_btn):
        if self.is_capturing:
            messagebox.showinfo("Busy", "Cannot test LEDs while capture is running.")
            return

        current = getattr(self, state_attr_name)

        if current:
            self.turn_off_all_leds()
            self.set_status("LEDs off")
        else:
            self.turn_off_all_leds()

            try:
                driver.on()
                target_led.on()
                setattr(self, state_attr_name, True)

                if target_btn is self.led1_btn:
                    label = "LED 1"
                elif target_btn is self.led2_btn:
                    label = "LED 2"
                elif target_btn is self.led3_btn:
                    label = "LED 3"
                else:
                    label = "LED 4"

                target_btn.config(bg="#4caf50", fg="white", text=f"{label} (ON)")
                self.set_status("LED test ON")

                self.set_tabs_for_led_test(True)
            except Exception as e:
                self.set_status(f"LED error: {e}")
                messagebox.showerror("LED Error", f"Failed to turn on LED: {e}")

    def on_reconnect_camera(self):
        if self.is_capturing:
            messagebox.showinfo("Busy", "Cannot reconnect cameras while capture is running.")
            return

        self.set_status("Reconnecting cameras...")
        self.update_idletasks()

        init_camera()
        init_usb_camera()

        if CAM_OK and USB_CAM_OK:
            self.set_status("FLIR + USB cameras connected")
            messagebox.showinfo(
                "Cameras",
                "Both cameras reconnected successfully.\n"
                "If lighting/tiles changed, run 'Calibrate Camera'."
            )
        else:
            errors = []
            if not CAM_OK:
                errors.append(f"FLIR: {CAM_ERROR_MSG}")
            if not USB_CAM_OK:
                errors.append(f"USB: {USB_CAM_ERROR_MSG}")

            messagebox.showerror("Camera error", "\n".join(errors))
            self.startup_camera_init()

    # =======================================================
    #  Capture Flow
    # =======================================================

    def startup_camera_init(self):
        self.set_status("Initializing cameras…")
        self.update_idletasks()

        init_camera()
        init_usb_camera()

        if not CAM_OK:
            messagebox.showerror(
                "FLIR camera error",
                f"Could not initialize FLIR camera:\n{CAM_ERROR_MSG}\n"
                "Check the connection and use Settings to reconnect."
            )
            return

        if not USB_CAM_OK:
            messagebox.showerror(
                "USB camera error",
                f"Could not initialize USB camera:\n{USB_CAM_ERROR_MSG}\n"
                "Check the connection and use Settings to reconnect."
            )
            return

        load_calibration_flats()
        if calibration_flats:
            self.set_status("FLIR + USB cameras initialized (calibration found).")
            self.start_btn.config(state="normal")
        else:
            self.set_status("FLIR + USB cameras initialized (no calibration).")
            self.calibrate_camera()

    def calibrate_camera(self):
        global calibration_flats

        if not CAM_OK:
            messagebox.showerror(
                "Camera Error",
                "FLIR camera is not connected or failed to initialize.\n"
                f"Details: {CAM_ERROR_MSG}"
            )
            return

        self.set_status("Waiting for user…")
        self.update_idletasks()

        messagebox.showinfo(
            "Prepare White Board",
            "Remove peanut tray and insert the full-size white calibration board.\n"
            "Click OK to continue."
        )

        self.set_status("Capturing calibration flats…")
        self.update_idletasks()

        leds_to_calibrate = [(1, led1), (2, led2), (3, led3)]
        calibration_flats = {}
        all_warnings = []

        try:
            for led_id, led_dev in leds_to_calibrate:
                self.set_status(f"Calibrating LED {led_id}…")
                self.update_idletasks()

                set_led_camera_cal_params(led_id)

                driver.on()
                led_dev.on()
                time.sleep(0.3)

                img = capture_image()

                led_dev.off()
                driver.off()
                time.sleep(0.2)

                if img is None:
                    all_warnings.append(f"LED {led_id}: Failed to capture calibration image.")
                    continue

                calibration_flats[led_id] = img

                save_calibration_flat(led_id, img, as_reference_if_missing=True)
                calib_raw_name = os.path.join(CALIB_DIR, f"LED{led_id}_CALIB_RAW.png")
                cv2.imwrite(calib_raw_name, img)

            self.set_status("Ready (calibrated)")
            self.start_btn.config(state="normal")

            if all_warnings:
                messagebox.showwarning(
                    "Calibration completed with warnings",
                    "\n".join(all_warnings)
                )
            else:
                messagebox.showinfo(
                    "Calibration",
                    "Calibration with white board completed."
                )

        except Exception as e:
            self.set_status(f"Calibration error: {e}")
            messagebox.showerror("Calibration error", str(e))

    def on_start_capture(self):
        if not CAM_OK:
            messagebox.showerror(
                "FLIR camera not ready",
                "FLIR camera not connected. Use Settings → Reconnect Cameras."
            )
            return

        if not USB_CAM_OK:
            messagebox.showerror(
                "USB camera not ready",
                "USB camera not connected. Use Settings → Reconnect Cameras."
            )
            return

        if not calibration_flats:
            if not messagebox.askokcancel(
                "No calibration",
                "No calibration data loaded.\n"
                "Capture anyway without flat-field correction?"
            ):
                return

        messagebox.showinfo(
            "Prepare Peanut Tray",
            "Insert the peanut tray with peanuts.\n"
            "Click OK to continue."
        )

        self.is_capturing = True
        self.start_btn.config(state="disabled")
        self.set_status("Starting capture...")
        self.set_progress(0.0)
        self.update_idletasks()

        self.capture_thread = threading.Thread(target=self.capture_sequence)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_sequence(self):
        try:
            # 3 FLIR captures + 1 USB capture
            total_steps = 4

            # ---------------------------------------------------
            # FLIR captures with LED1-3
            # ---------------------------------------------------
            leds = [(1, led1), (2, led2), (3, led3)]
            band_imgs = {}
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            for idx, (i, led_dev) in enumerate(leds, start=1):
                self.safe_status(f"Capturing LED {i}...")
                set_led_camera_params(i)

                driver.on()
                led_dev.on()
                time.sleep(0.3)

                img = capture_image()

                led_dev.off()
                driver.off()
                time.sleep(0.2)

                if img is None:
                    print(f"[LED {i}] Failed to capture image.")
                    continue

                img_norm, norm_ratio = flat_field_normalize(img, i)

                raw_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED{i}_raw_.png")
                norm_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED{i}_norm.png")
                norm_ratio_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED{i}_ratio.npy")

                band_imgs[i] = img_norm
                cv2.imwrite(raw_name, img)
                cv2.imwrite(norm_name, img_norm)

                if norm_ratio is not None:
                    np.save(norm_ratio_name, norm_ratio.astype(np.float32))

                progress = idx / total_steps * 100.0
                self.safe_progress(progress)
            
            # Build 3-channel cube: [LED1/405, LED2/720, LED3/760].
            # This is the input used by the YOLO + PCA/regression maturity analysis.
            missing = [i for i in (1, 2, 3) if i not in band_imgs]
            if missing:
                raise RuntimeError(f"Missing captured band(s): {missing}")

            cube = np.dstack([band_imgs[1], band_imgs[2], band_imgs[3]])
            cube_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED123_cube.npy")
            pseudo_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED123_pseudo.png")
            np.save(cube_name, cube.astype(np.float32))
            cv2.imwrite(pseudo_name, cube)

            # ---------------------------------------------------
            # Run segmentation + maturity analysis on LED1-3 cube
            # ---------------------------------------------------
            try:
                self.safe_status("Analyzing peanut maturity...")
                load_analysis_models()

                result = process_one_cube(
                    data=cube.astype(np.float32),
                    stem=f"{timestamp}_LED123_cube",
                    pca_model=pca_model,
                    reg_model=reg_model,
                    yolo_model=yolo_model,
                    out_dir=ANALYSIS_OUTPUT_DIR,
                    normalize_pixels=True,
                    device=ANALYSIS_DEVICE,
                    conf=ANALYSIS_CONF,
                    iou_thresh=ANALYSIS_IOU_THRESH,
                    imgsz=ANALYSIS_IMGSZ,
                    max_det=ANALYSIS_MAX_DET,
                    area_min=ANALYSIS_AREA_MIN,
                )
                self.safe_update_results(result)
                print("[Analysis] Result:", result)

            except Exception as e:
                # Capture should still continue even if analysis models are missing/fail.
                print("[Analysis] Skipped/failed:", e)
                self.safe_status(f"Capture OK, analysis skipped: {e}")

            # ---------------------------------------------------
            # USB reference capture with LED4
            # ---------------------------------------------------
            self.safe_status("Capturing LED 4 reference image...")

            driver.on()
            led4.on()
            time.sleep(1)

            usb_img = capture_usb_image()
            print("[LED4] frame mean =", float(usb_img.mean()))

            led4.off()
            driver.off()
            time.sleep(0.2)
            
            led4_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED4.png")
            cv2.imwrite(led4_name, usb_img)

            self.safe_progress(100.0)
            self.safe_status("Capture complete.")
            self.safe_refresh_gallery()

        except Exception as e:
            print("[Capture] Error:", e)
            self.safe_status(f"Error: {e}")

        finally:
            self.safe_capture_end()

    # =======================================================
    #  Gallery
    # =======================================================

    def load_image_list(self):
        self.image_listbox.delete(0, tk.END)

        if not os.path.exists(IMAGE_DIR):
            return

        files = [
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith(".png")
        ]

        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(IMAGE_DIR, f)),
            reverse=True
        )

        for f in files:
            self.image_listbox.insert(tk.END, f)

    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return

        filename = self.image_listbox.get(selection[0])
        filepath = os.path.join(IMAGE_DIR, filename)
        self.show_preview(filepath)

    def show_preview(self, filepath: str):
        if not os.path.exists(filepath):
            return

        try:
            img = Image.open(filepath)

            if img.mode == "I;16":
                arr = np.array(img, dtype=np.uint16)
                arr8 = (arr / 256).astype("uint8")
                img = Image.fromarray(arr8, mode="L")

            max_w, max_h = 520, 400
            img.thumbnail((max_w, max_h))

            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_img, text="")
        except Exception as e:
            self.preview_label.config(text=f"Error loading image:\n{e}")
            self.preview_img = None

    # =======================================================
    #  Closing
    # =======================================================

    def on_close(self):
        if self.is_capturing:
            if not messagebox.askokcancel(
                "Quit",
                "Capture in progress. Do you really want to quit?"
            ):
                return

        cleanup_hardware()
        self.destroy()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    try:
        app = PeanutApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        cleanup_hardware()