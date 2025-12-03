#!/usr/bin/env python3
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

if platform.system() == "Linux":
    from gpiozero import OutputDevice
else:
    # Fake OutputDevice for Windows so code doesn't crash
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            print("[MOCK] OutputDevice created (Windows)")
        def on(self):  print("[MOCK] ON")
        def off(self): print("[MOCK] OFF")


print(platform.system())
print(platform.release())
print(platform.version())
print(platform.platform())
# ============================================================
#  CONFIG
# ============================================================

# Directory to save images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"   # Force a working GPIO backend (must be set before creating OutputDevice)

# Relay pin definitions (BCM)
DRIVER_PIN = 17
LED1_PIN   = 27
LED2_PIN   = 22
LED3_PIN   = 23

# --- Initialize relays (active-LOW: LOW = ON, HIGH = OFF) ---
print("[Init] Setting up GPIO...")
driver = OutputDevice(DRIVER_PIN, active_high=False, initial_value=False)
led1   = OutputDevice(LED1_PIN,   active_high=False, initial_value=False)
led2   = OutputDevice(LED2_PIN,   active_high=False, initial_value=False)
led3   = OutputDevice(LED3_PIN,   active_high=False, initial_value=False)

CAM_OK = False
CAM_ERROR_MSG = ""
system = None
cam_list = None
cam = None
processor = None

# --- Reference ROI ---
WHITE_ROIS = [
    (348,950,500,1189),   
    (1641,199,1836,462), 
]
BLACK_ROIS = [
    (333,195,511,436),   
    (1654,921,1821,1173), 
]

# QC thresholds 
MAX_VAL = 255 #255.0   # Mono16, Mono8
TARGET_WHITE = 180 #180.0   # target mean for white in calibration (0-255)
TARGET_BLACK = 20 #20.0    # target mean for black in calibration
WHITE_TOL = 5 #5.0        # +/- range for white during calibration
DR_MIN = 30 #30.0          # minimum dynamic range (I_white - I_black)
SAT_THRESH = 0.98      # fraction of MAX_VAL considered "too close to saturation"
STD_WHITE_MAX = 8 #8.0    # if std of white patch > this, warn (dirty/glare)
STD_BLACK_MAX = 8 #8.0    # if std of black patch > this, warn
DRIFT_FRAC_MAX = 0.10  # 10% drift allowed vs calibration

EXP_MIN = None
EXP_MAX = None
GAIN_MIN = None
GAIN_MAX = None

calibration_results = {}
itertaions = 5

def reset_camera():
    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor

    if cam is not None:
        cam.DeInit()
    if cam_list is not None:
        cam_list.Clear()
    if system is not None:
        system.ReleaseInstance()

    processor = None
    CAM_OK = False
    CAM_ERROR_MSG = ""

def init_camera():
    print("[Init] Setting up FLIR camera ...")

    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor, EXP_MAX, EXP_MIN, GAIN_MAX, GAIN_MIN
    CAM_OK = False
    CAM_ERROR_MSG = ""
    system = cam_list = cam = processor = None

    reset_camera()

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() == 0:
            CAM_ERROR_MSG = "No FLIR camera found"
            print("[Init] No cameras detected.")
            return

        cam = cam_list.GetByIndex(0)
        cam.Init()
        CAM_OK = True

        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)  
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.ExposureTime.SetValue(17800.0)  # microseconds
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        cam.Gain.SetValue(46.6)
        processor = PySpin.ImageProcessor()

        EXP_MIN =  cam.ExposureTime.GetMin()
        EXP_MAX =  cam.ExposureTime.GetMax()
        GAIN_MIN = cam.Gain.GetMin()    
        GAIN_MAX = cam.Gain.GetMax()

        print(f"Camera exposure range: {EXP_MIN} to {EXP_MAX} us")
        print(f"Camera gain range: {GAIN_MIN} to {GAIN_MAX} dB")
        print("[Init] Camera Init OK")

    except Exception as e:
        CAM_ERROR_MSG = f"Camera init error: {e!r}"
        CAM_OK = False
        print(CAM_ERROR_MSG)
        return

def capture_image():
    global CAM_OK, CAM_ERROR_MSG

    if not CAM_OK or cam is None or processor is None:
        raise RuntimeError("Camera not initialized")

    try:
        cam.BeginAcquisition()
        img = cam.GetNextImage(1000)
    except Exception as e:
        CAM_OK = False
        CAM_ERROR_MSG = f"Acquisition error: Check the camera connection"
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
    return arr

def validate_rois(img_shape):
    """Check that ROIs are inside image bounds."""
    h, w = img_shape[:2]
    for roi in WHITE_ROIS + BLACK_ROIS:
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(
                f"ROI {roi} is out of image bounds (w={w}, h={h}). "
            )
        
def roi_stats(img, roi_list):
    """Return mean and std across all pixels in given list of ROIs."""
    means = []
    stds = []
    for (x1, y1, x2, y2) in roi_list:
        patch = img[y1:y2, x1:x2]
        #print(f"ROI {(x1, y1, x2, y2)} -> patch shape {patch.shape}")
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
    
    return warnings

def calibrate_led(led_id, led_device):
    global CAM_OK, CAM_ERROR_MSG
    print(f"\n=== Calibration for LED {led_id} ===")

    try:
        exp_us = cam.ExposureTime.GetValue()
    except Exception as e:
        # Camera likely unplugged or failed mid-run
        CAM_OK = False
        CAM_ERROR_MSG = f"Acquisition error: {e!r}"
        print("[Camera] Exposure aquisition failed:", CAM_ERROR_MSG)
        raise RuntimeError(CAM_ERROR_MSG)
    
    print(f"  Starting exposure: {exp_us:.1f} us")

    for iteration in range(itertaions):  # up to 12 iterations
        driver.on()
        led_device.on()
        time.sleep(0.3)

        img = capture_image()

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
        if Iw > SAT_THRESH * MAX_VAL:
            exp_us *= 0.7
        elif Iw > TARGET_WHITE:
            exp_us *= 0.9
        else:
            exp_us *= 1.3

        # Clamp exposure
        exp_us = max(EXP_MIN, min(EXP_MAX, exp_us))
        cam.ExposureTime.SetValue(exp_us)
    led_device.off()
    driver.off()
    time.sleep(0.3)
    print("  -> Calibration loop ended without perfect convergence.")
    return exp_us, Iw, Ib

def cleanup_hardware():
    print("[Cleanup] Releasing hardware...")
    # GPIO
    for dev in [driver, led1, led2, led3]:
        try:
            dev.off()
            dev.close()
        except Exception:
            pass

    # Camera
    try:
        if CAM_OK and cam is not None:
            cam.DeInit()
        if cam_list is not None:
            cam_list.Clear()
        if system is not None:
            system.ReleaseInstance()
    except Exception:
        pass

    print("[Cleanup] GPIO and camera released.")


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
            self.attributes("-fullscreen",True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.capture_thread = None
        self.is_capturing = False
        self.preview_img = None  # keep reference to avoid GC
        self.led1_on = False
        self.led2_on = False
        self.led3_on = False

        # Tk variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var   = tk.StringVar(value="Idle")

        self._create_style()
        self._create_widgets()

        self.start_btn.config(state="disabled")
        self.set_status("Initializing camera…")
        self.after(300, self.startup_camera_init)

    # ---------------- Styles ----------------
    def _create_style(self):
        style = ttk.Style(self)
        style.configure("TButton", font=("Helvetica", 14))
        style.configure("TLabel",  font=("Helvetica", 12))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Start.TButton",font=("Helvetica", 36, "bold"),borderwidth=0,focuscolor="",padding=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 14), padding=[10, 5],)
        style.configure("LedOff.TButton",font=("Helvetica", 12),padding=5)
        style.configure("LedOn.TButton",font=("Helvetica", 12, "bold"),padding=5,background="#4caf50",  foreground="white")
        style.map("LedOn.TButton",background=[("active", "#66bb6a")])

    # ---------------- Layout ----------------
    def _create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # ==== Capture tab ====
        self.tab_capture = ttk.Frame(notebook)
        notebook.add(self.tab_capture, text="Capture")

        self.notebook = notebook   # keep reference if you want

        self.tab_capture.columnconfigure(0, weight=1)   
        self.tab_capture.columnconfigure(1, weight=2)   
        self.tab_capture.rowconfigure(0, weight=1)

        # ------ Left side: Start button ------
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

        # Progress bar + status below (spanning both columns)
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
        self.status_label.grid(row=2, column=0, columnspan=2,
                               pady=(0, 10))

        # ------ Right side: Results panel ------
        right_cap = ttk.LabelFrame(self.tab_capture, text="Latest Results")
        right_cap.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        right_cap.columnconfigure(0, weight=1)
        for r in range(6):
            right_cap.rowconfigure(r, weight=1)

        # StringVars for results (placeholders for now)
        self.total_var  = tk.StringVar(value="Total peanuts: -")
        self.black_var  = tk.StringVar(value="Black: -")
        self.brown_var  = tk.StringVar(value="Brown: -")
        self.yellow_var = tk.StringVar(value="Yellow: -")
        self.white_var  = tk.StringVar(value="White: -")

        ttk.Label(right_cap, textvariable=self.total_var).grid(
            row=0, column=0, sticky="w", padx=10, pady=2
        )
        ttk.Label(right_cap, textvariable=self.black_var).grid(
            row=1, column=0, sticky="w", padx=10, pady=2
        )
        ttk.Label(right_cap, textvariable=self.brown_var).grid(
            row=2, column=0, sticky="w", padx=10, pady=2
        )
        ttk.Label(right_cap, textvariable=self.yellow_var).grid(
            row=3, column=0, sticky="w", padx=10, pady=2
        )
        ttk.Label(right_cap, textvariable=self.white_var).grid(
            row=4, column=0, sticky="w", padx=10, pady=2
        )

        # ==== Gallery tab ====
        self.tab_gallery = ttk.Frame(notebook)
        notebook.add(self.tab_gallery, text="Gallery")

        # 30% list / 70% preview
        self.tab_gallery.columnconfigure(0, weight=1)
        self.tab_gallery.columnconfigure(1, weight=1)
        self.tab_gallery.rowconfigure(0, weight=1)

        # Left side
        left_frame = ttk.Frame(self.tab_gallery)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        list_header = ttk.Label(
            left_frame, text="Captured Images", style="Header.TLabel"
        )
        list_header.grid(row=0, column=0, pady=(0, 5))

        self.image_listbox = tk.Listbox(
            left_frame, width=30, font=("Helvetica", 11)
        )
        self.image_listbox.grid(row=1, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        self.refresh_btn = ttk.Button(
            left_frame, text="Refresh List", command=self.load_image_list
        )
        self.refresh_btn.grid(row=2, column=0, pady=(5, 0), sticky="ew")

        # Right side
        right_frame = ttk.Frame(self.tab_gallery)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        center_frame = ttk.Frame(right_frame)
        center_frame.grid(row=0, column=0)
        

        self.preview_label = ttk.Label(center_frame, text="No image selected")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Initial list load
        self.load_image_list()

        # ===== Settings tab =====
        self.tab_settings = ttk.Frame(notebook)
        notebook.add(self.tab_settings, text="Settings")

        self.tab_settings.columnconfigure(0, weight=1)
        for r in range(4):
            self.tab_settings.rowconfigure(r, weight=1)


        # ---- Manual LED test buttons ----
        leds_frame = ttk.LabelFrame(self.tab_settings, text="Manual LED Test")
        leds_frame.grid(row=1, column=0, pady=10, padx=40, sticky="ew")
        for c in range(3):
            leds_frame.columnconfigure(c, weight=1)

        self.led1_btn = tk.Button(
            leds_frame,
            text="LED 1",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            command=lambda: self.toggle_led_exclusive(led1, "led1_on", self.led1_btn)
        )
        self.led1_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.led2_btn = tk.Button(
            leds_frame,
            text="LED 2",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            command=lambda: self.toggle_led_exclusive(led2, "led2_on", self.led2_btn)
        )
        self.led2_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.led3_btn = tk.Button(
            leds_frame,
            text="LED 3",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            command=lambda: self.toggle_led_exclusive(led3, "led3_on", self.led3_btn)
        )
        self.led3_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.led_default_bg = self.led1_btn.cget("bg")
        self.led_default_fg = self.led1_btn.cget("fg")
        self.led_default_bg = self.led2_btn.cget("bg")
        self.led_default_fg = self.led2_btn.cget("fg")
        self.led_default_bg = self.led3_btn.cget("bg")
        self.led_default_fg = self.led3_btn.cget("fg")


        # ---- Camera controls ----
        cam_frame = ttk.LabelFrame(self.tab_settings, text="Camera")
        cam_frame.grid(row=2, column=0, pady=10, padx=40, sticky="ew")
        cam_frame.columnconfigure(0, weight=1)
        cam_frame.columnconfigure(1, weight=1)

        reconnect_btn = ttk.Button(
            cam_frame, text="Reconnect Camera",
            command=self.on_reconnect_camera
        )
        reconnect_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        calibrate_btn = ttk.Button(
            cam_frame,
            text="Calibrate Camera",
            command=self.calibrate_camera
        )
        calibrate_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # ---- Exit button ----
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
        self.status_label.grid(row=4, column=0, columnspan=2,
                               pady=(0, 10))

    # =======================================================
    #  Status / Progress helpers
    # =======================================================

    def set_status(self, text: str):
        """Update status label (main thread)."""
        self.status_var.set(text)

    def set_progress(self, value: float):
        """Update progress bar (main thread)."""
        self.progress_var.set(value)

    def safe_status(self, text: str):
        """Thread-safe status update from worker thread."""
        self.after(0, lambda: self.set_status(text))

    def safe_progress(self, value: float):
        """Thread-safe progress update from worker thread."""
        self.after(0, lambda: self.set_progress(value))

    def safe_capture_end(self):
        """Thread-safe end-of-capture state reset."""
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
        driver.off()

        # Reset state flags
        self.led1_on = self.led2_on = self.led3_on = False

        # Reset button appearance if buttons already exist
        if hasattr(self, "led1_btn"):
            self.led1_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 1")
        if hasattr(self, "led2_btn"):
            self.led2_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 2")
        if hasattr(self, "led3_btn"):
            self.led3_btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text="LED 3")

        self.set_tabs_for_led_test(False)

    def toggle_led_exclusive(self, target_led, state_attr_name, target_btn):
        if self.is_capturing:
            messagebox.showinfo(
                "Busy",
                "Cannot test LEDs while capture is running."
            )
            return

        current = getattr(self, state_attr_name)

        if current:
            # LED is ON -> turn everything OFF
            self.turn_off_all_leds()
            self.set_status("LEDs off")
        else:
            # Make sure only this one is ON
            self.turn_off_all_leds()

            try:
                driver.on()
                target_led.on()
                setattr(self, state_attr_name, True)

                if target_btn is self.led1_btn:
                    label = "LED 1"
                elif target_btn is self.led2_btn:
                    label = "LED 2"
                else:
                    label = "LED 3"

                target_btn.config(bg="#4caf50", fg="white", text=f"{label} (ON)")
                self.set_status("LED test ON")

                self.set_tabs_for_led_test(True)
            except Exception as e:
                self.set_status(f"LED error: {e}")
                messagebox.showerror("LED Error", f"Failed to turn on LED: {e}")

    def on_reconnect_camera(self):
        """Try to reinitialize the camera from Settings tab."""

        if self.is_capturing:
            messagebox.showinfo(
                "Busy",
                "Cannot reconnect camera while capture is running."
            )
            return

        self.set_status("Reconnecting camera...")
        self.update_idletasks()

        init_camera()

        if CAM_OK:
            self.set_status("Camera connected (not calibrated)")
            messagebox.showinfo(
                "Camera",
                "Camera reconnected successfully.\n"
                "If lighting/tiles changed, run 'Calibrate LEDs'."
            )
        else:
            self.startup_camera_init()  


    # =======================================================
    #  Capture Flow
    # =======================================================

    def startup_camera_init(self):
        """Run at startup: initialize camera and calibrate LEDs (tray empty)."""
        self.set_status("Initializing camera…")
        self.update_idletasks()

        init_camera()

        if not CAM_OK:
            messagebox.showerror(
                "Camera error",
                f"Could not initialize camera:\n{CAM_ERROR_MSG} - check the connection and use Settings to reconnect"
            )
            return
        
        self.set_status("Camera Initialized")
        self.update_idletasks()

        self.calibrate_camera()

    def calibrate_camera(self):
        global calibration_results

        if not CAM_OK:
            messagebox.showerror(
                "Camera Error",
                "Camera is not connected or failed to initialize.\n"
                f"Details: {CAM_ERROR_MSG}"
            )
            return
        
        self.set_status("Waiting for user…")
        self.update_idletasks()

        messagebox.showinfo(
        "Prepare Tray",
        "Remove peanuts and insert reference tray.\nClick OK to continue."
        )
        
        self.set_status("Calibrating LEDs…")
        self.update_idletasks()

        leds_to_calibrate = [(1, led1),(2, led2),(3, led3)]
        calibration_results = {}

        try:
            for led_id, led_dev in leds_to_calibrate:
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
            self.set_status("Ready (calibrated)")
            messagebox.showinfo(
                "Camera",
                "Camera calibrated sucessfully."
            )
            self.start_btn.config(state="normal")

        except Exception as e:
            self.set_status(f"Calibration error: {e}")
            messagebox.showerror("Calibration error", str(e))

    def on_start_capture(self):
        if not CAM_OK:
            messagebox.showerror(
                "Camera not ready",
                "Camera not connected. Use Settings → Reconnect Camera."
            )
            return

        self.is_capturing = True
        self.start_btn.config(state="disabled")
        self.set_status("Starting capture...")
        self.set_progress(0.0)
        self.update_idletasks()

        self.capture_thread = threading.Thread(target=self.capture_sequence)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_sequence(self):
        global calibration_results, CAM_OK, CAM_ERROR_MSG

        try:
            leds = [(1, led1), (2, led2), (3, led3)]
            total_steps = len(leds)

            for idx, (i, led) in enumerate(leds, start=1):
                self.safe_status(f"Capturing LED {i}...")

                if i in calibration_results:
                    try:
                        cam.ExposureTime.SetValue(calibration_results[i]["exposure_us"])
                    except Exception as e:
                        CAM_OK = False
                        CAM_ERROR_MSG = f"Acquisition error: Setting exposure failed"
                        print("[Camera] Exposure aquisition failed:", CAM_ERROR_MSG)
                        raise RuntimeError(CAM_ERROR_MSG)
    
                driver.on()
                led.on()
                time.sleep(0.3)

                img = capture_image()

                led.off()
                driver.off()
                time.sleep(0.2)

                if img is None:
                    print(f"[LED {i}] Failed to capture image.")
                    continue

                Iw, std_w = roi_stats(img, WHITE_ROIS)
                Ib, std_b = roi_stats(img, BLACK_ROIS)

                print(
                    f"[LED {i}] Capture stats: Iw={Iw:.1f}, Ib={Ib:.1f}, "
                    f"std_w={std_w:.1f}, std_b={std_b:.1f}"
                )

                cal = calibration_results.get(i, None)
                if cal is not None:
                    cal_Iw = cal["I_white"]
                    cal_Ib = cal["I_black"]
                else:
                    cal_Iw, cal_Ib = Iw, Ib  # fallback

                warnings = apply_qc_and_print(i, Iw, Ib, std_w, std_b, cal_Iw, cal_Ib)

                if warnings:
                    full_msg = "\n".join(warnings)

                    # Suggest user actions
                    suggestion = (
                        "\n\nSuggestions:\n"
                        "- Clean the white and black reference tiles.\n"
                        "- Ensure tray is fully inside and flat.\n"
                        "- Check for external light leaking into imaging box.\n"
                        "- Check LEDs for dirt or misalignment.\n"
                        "- Recalibrate if necessary (Settings → Recalibrate LEDs)."
                    )

                    full_msg += suggestion

                    # Show pop-up on GUI thread
                    self.safe_status("QC Warning")
                    self.after(0, lambda: messagebox.showwarning(
                        f"QC Warning - LED {i}", full_msg
                    ))

                # Normalize and save
                img_norm = normalize_with_refs(img, Iw, Ib)          # 0..1
                img_norm_8u = (img_norm * 255.0).astype("uint8")     # viewable

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                raw_name = os.path.join(IMAGE_DIR, f"LED{i}_raw_{timestamp}.png")
                norm_name = os.path.join(IMAGE_DIR, f"LED{i}_norm_{timestamp}.png")

                cv2.imwrite(raw_name, img)          # 16-bit PNG
                cv2.imwrite(norm_name, img_norm_8u) # 8-bit PNG

                print(f"[LED {i}] Saved raw -> {raw_name}")
                print(f"[LED {i}] Saved normalized -> {norm_name}")

                progress = idx / total_steps * 100.0
                self.safe_progress(progress)

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
        """Load image filenames into listbox, sorted by date (newest first)."""
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
        """Handle image selection from listbox and show preview."""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        filename = self.image_listbox.get(selection[0])
        filepath = os.path.join(IMAGE_DIR, filename)
        self.show_preview(filepath)

    def show_preview(self, filepath: str):
        """Display selected image on the right side."""
        if not os.path.exists(filepath):
            return

        try:
            img = Image.open(filepath)

            if img.mode == "I;16":  
                arr = np.array(img, dtype=np.uint16)
                arr8 = (arr / 256).astype("uint8")   # 12-bit/16-bit → 8-bit
                img = Image.fromarray(arr8, mode="L")
            max_w, max_h = 520, 400  # fits in 800x480 layout
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
