#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import PySpin
from gpiozero import OutputDevice
from datetime import datetime

# Force a working backend (lgpio or pigpio)
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"   # or "pigpio" if preferred

# --- Relay pin definitions (BCM) ---
DRIVER = 17
LED1   = 27
LED2   = 22
LED3   = 23

# --- Initialize relays (active-LOW: LOW = ON, HIGH = OFF) ---
driver = OutputDevice(DRIVER, active_high=False, initial_value=False)
led1   = OutputDevice(LED1,   active_high=False, initial_value=False)
led2   = OutputDevice(LED2,   active_high=False, initial_value=False)
led3   = OutputDevice(LED3,   active_high=False, initial_value=False)

# --- Camera setup ---
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
if cam_list.GetSize() == 0:
    print("No FLIR camera found.")
    cam_list.Clear()
    system.ReleaseInstance()
    exit(1)

cam = cam_list.GetByIndex(0)
cam.Init()

nodemap = cam.GetNodeMap()

# Disable auto exposure and gain
exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
exp_auto_off = exp_auto.GetEntryByName("Off")
exp_auto.SetIntValue(exp_auto_off.GetValue())

gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
gain_auto_off = gain_auto.GetEntryByName("Off")
gain_auto.SetIntValue(gain_auto_off.GetValue())

# Manual exposure/gain nodes
exp_time_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))

# Start from some safe defaults
exp_time_node.SetValue(5000.0)  # microseconds
gain_node.SetValue(0.0)         # keep gain at 0 for now

EXP_MIN = exp_time_node.GetMin()
EXP_MAX = exp_time_node.GetMax()

# --- Capture directory ---
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# --- Reference ROI definitions ---
# IMPORTANT: UPDATE THESE FOR YOUR IMAGE GEOMETRY.
# Each ROI is a tuple: (x1, y1, x2, y2)
# Two white tiles, two black tiles (e.g., in 4 corners but *not* at extreme edges).
WHITE_ROIS = [
    (100, 100, 150, 150),   # top-left white tile (example)
    (1000, 100, 1050, 150), # top-right white tile (example)
]
BLACK_ROIS = [
    (100, 800, 150, 850),   # bottom-left black tile (example)
    (1000, 800, 1050, 850), # bottom-right black tile (example)
]

# QC thresholds (tune these after some runs)
MAX_VAL = 255.0   # Mono8
TARGET_WHITE = 180.0   # target mean for white in calibration (0-255)
TARGET_BLACK = 20.0    # target mean for black in calibration
WHITE_TOL = 5.0        # +/- range for white during calibration
DR_MIN = 30.0          # minimum dynamic range (I_white - I_black)
SAT_THRESH = 0.98      # fraction of MAX_VAL considered "too close to saturation"
STD_WHITE_MAX = 8.0    # if std of white patch > this, warn (dirty/glare)
STD_BLACK_MAX = 8.0    # if std of black patch > this, warn

DRIFT_FRAC_MAX = 0.10  # 10% drift allowed vs calibration

# --- Helper functions ---

def capture_array():
    """Capture one image from the camera and return as np.ndarray (Mono8)."""
    cam.BeginAcquisition()
    processor = PySpin.ImageProcessor()
    img = cam.GetNextImage(1000)
    if img.IsIncomplete():
        print("[x] Incomplete image.")
        img.Release()
        cam.EndAcquisition()
        return None
    arr = processor.Convert(img, PySpin.PixelFormat_Mono8).GetNDArray()
    img.Release()
    cam.EndAcquisition()
    return arr

def validate_rois(img_shape):
    """Check that ROIs are inside image bounds."""
    h, w = img_shape[:2]
    for roi in WHITE_ROIS + BLACK_ROIS:
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(
                f"ROI {roi} is out of image bounds (w={w}, h={h}). "
                "Update WHITE_ROIS / BLACK_ROIS to match your tiles."
            )

def roi_stats(img, roi_list):
    """Return mean and std across all pixels in given list of ROIs."""
    means = []
    stds = []
    for (x1, y1, x2, y2) in roi_list:
        patch = img[y1:y2, x1:x2]
        means.append(patch.mean())
        stds.append(patch.std())
    return float(np.mean(means)), float(np.mean(stds))

def normalize_with_refs(img, I_white, I_black):
    """Normalize image to [0,1] using current white/black intensities."""
    eps = 1e-6
    gain = 1.0 / max(I_white - I_black, eps)
    offset = -I_black * gain
    img_norm = img.astype(np.float32) * gain + offset
    img_norm = np.clip(img_norm, 0.0, 1.0)
    return img_norm

def apply_qc_and_print(led_id, Iw, Ib, std_w, std_b, cal_Iw, cal_Ib):
    """Check various QC conditions and print warnings."""
    dyn_range = Iw - Ib
    warnings = []

    # Drift relative to calibration
    if cal_Iw > 1e-3:
        drift_white = abs(Iw - cal_Iw) / cal_Iw
        if drift_white > DRIFT_FRAC_MAX:
            warnings.append(
                f"White drift {drift_white*100:.1f}% vs calibration. "
                "Lighting/exposure changed."
            )

    # Saturation
    if Iw > SAT_THRESH * MAX_VAL:
        warnings.append("White reference near saturation. Reduce exposure/gain.")

    # Dynamic range too small
    if dyn_range < DR_MIN:
        warnings.append(
            f"Dynamic range too low (Iw - Ib = {dyn_range:.1f}). "
            "Lighting too weak or references too similar."
        )

    # Std checks
    if std_w > STD_WHITE_MAX:
        warnings.append(
            f"White patch std {std_w:.1f} too high. "
            "Tile may be dirty or has glare/shadow."
        )
    if std_b > STD_BLACK_MAX:
        warnings.append(
            f"Black patch std {std_b:.1f} too high. "
            "Stray light or contamination on black tile."
        )

    if warnings:
        print(f"[LED {led_id}] WARNINGS:")
        for wmsg in warnings:
            print("   -", wmsg)
    else:
        print(f"[LED {led_id}] QC OK.")

def calibrate_led(led_id, led_device):
    """
    Calibration step for a single LED, without peanuts:
    - Adjust exposure until white and black reach target ranges.
    - Return: calibrated_exposure_us, I_white_cal, I_black_cal
    """
    print(f"\n=== Calibration for LED {led_id} ===")
    # Simple iterative adjustment of exposure only (gain fixed at 0)
    exp_us = exp_time_node.GetValue()
    print(f"  Starting exposure: {exp_us:.1f} us")

    for iteration in range(12):  # up to 12 iterations
        driver.on()
        led_device.on()
        time.sleep(0.3)

        img = capture_array()
        led_device.off()
        driver.off()

        if img is None:
            print("  Failed to capture image during calibration.")
            continue

        if iteration == 0:
            validate_rois(img.shape)

        Iw, std_w = roi_stats(img, WHITE_ROIS)
        Ib, std_b = roi_stats(img, BLACK_ROIS)
        dyn_range = Iw - Ib

        print(
            f"  Iter {iteration}: Iw={Iw:.1f}, Ib={Ib:.1f}, "
            f"std_w={std_w:.1f}, std_b={std_b:.1f}, exp={exp_us:.1f} us"
        )

        # Check if within acceptable calibration range
        if (
            abs(Iw - TARGET_WHITE) <= WHITE_TOL
            and dyn_range >= DR_MIN
            and Iw < SAT_THRESH * MAX_VAL
        ):
            print("  -> Calibration target reached.")
            return exp_us, Iw, Ib

        # Decide how to tweak exposure
        if Iw > TARGET_WHITE or Iw > SAT_THRESH * MAX_VAL:
            # too bright or near saturation → decrease exposure
            exp_us *= 0.7
        else:
            # too dark or insufficient dynamic range → increase exposure
            exp_us *= 1.3

        # Clamp exposure
        exp_us = max(EXP_MIN, min(EXP_MAX, exp_us))
        exp_time_node.SetValue(exp_us)

    print("  -> Calibration loop ended without perfect convergence.")
    # Return last values anyway
    return exp_us, Iw, Ib

# --- Main LED sequence with calibration + capture ---

try:
    leds = [
        (1, led1),
        (2, led2),
        (3, led3),
    ]

    # 1) Calibration phase (NO PEANUTS in tray)
    print("\n=== STEP 1: Calibration ===")
    print("Make sure ONLY white/black reference tiles are visible (no peanuts).")
    input("Press Enter to start calibration...")

    calibration_results = {}  # led_id -> dict

    for led_id, led_dev in leds:
        calib_exp, calib_Iw, calib_Ib = calibrate_led(led_id, led_dev)
        calibration_results[led_id] = {
            "exposure_us": calib_exp,
            "I_white": calib_Iw,
            "I_black": calib_Ib,
        }
        print(
            f"[LED {led_id}] Calibrated: exp={calib_exp:.1f} us, "
            f"Iw={calib_Iw:.1f}, Ib={calib_Ib:.1f}"
        )

    print("\n=== STEP 2: Capture with peanuts ===")
    print("Now place peanuts on the tray (reference tiles must still be visible).")
    input("Press Enter to capture images...")

    for led_id, led_dev in leds:
        print(f"\n--> Capturing LED {led_id}")
        # Use calibrated exposure
        exp_time_node.SetValue(calibration_results[led_id]["exposure_us"])
        gain_node.SetValue(0.0)  # still zero; you can make this per-LED if needed

        driver.on()
        time.sleep(0.2)
        led_dev.on()
        time.sleep(1.0)  # allow lighting to stabilize

        img = capture_array()

        led_dev.off()
        driver.off()

        if img is None:
            print(f"[LED {led_id}] Failed to capture image.")
            continue

        # Stats for this capture
        Iw, std_w = roi_stats(img, WHITE_ROIS)
        Ib, std_b = roi_stats(img, BLACK_ROIS)
        print(
            f"[LED {led_id}] Capture stats: Iw={Iw:.1f}, Ib={Ib:.1f}, "
            f"std_w={std_w:.1f}, std_b={std_b:.1f}"
        )

        cal_Iw = calibration_results[led_id]["I_white"]
        cal_Ib = calibration_results[led_id]["I_black"]

        # QC / warnings
        apply_qc_and_print(led_id, Iw, Ib, std_w, std_b, cal_Iw, cal_Ib)

        # Normalize image using current white/black
        img_norm = normalize_with_refs(img, Iw, Ib)       # 0..1
        img_norm_8u = (img_norm * 255.0).astype(np.uint8) # for saving

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        raw_name = os.path.join(CAPTURE_DIR, f"LED{led_id}_raw_{timestamp}.png")
        norm_name = os.path.join(CAPTURE_DIR, f"LED{led_id}_norm_{timestamp}.png")

        cv2.imwrite(raw_name, img)
        cv2.imwrite(norm_name, img_norm_8u)

        print(f"[LED {led_id}] Saved raw -> {raw_name}")
        print(f"[LED {led_id}] Saved normalized -> {norm_name}")

    print("\nAll captures complete!")

except KeyboardInterrupt:
    print("\n[!] Interrupted by user.")
finally:
    # Turn everything off and release
    for dev in [driver, led1, led2, led3]:
        try:
            dev.off()
            dev.close()
        except Exception:
            pass

    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    print("GPIO and camera released.")
