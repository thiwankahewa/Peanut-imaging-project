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
    # Fake OutputDevice for Windows 
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            print("[MOCK] OutputDevice created (Windows)")
        def on(self):  print("[MOCK] ON")
        def off(self): print("[MOCK] OFF")

# ============================================================
#  CONFIG
# ============================================================

# Directory to save images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)
CALIB_DIR = "calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"   # Force a working GPIO backend 

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

TRAY_ROI = (151,234,1759,1285)
DOOR_ROI = (1643, 1237, 1753, 1343)
DOOR_MARKER_ID = 0
DOOR_MIN_AREA = 2000

LED_EXPOSURE_US = {1: 17500.0, 2: 17500.0, 3: 17000.0,}
LED_GAIN_DB = {1: 8.0, 2: 2.0, 3: 0.0,}
calibration_flats = {}

def reset_camera():
    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor

    if cam is not None:
        cam.DeInit()
    if cam_list is not None:
        cam_list.Clear()
    if system is not None:
        system.ReleaseInstance()

    CAM_OK = False
    CAM_ERROR_MSG = ""
    system = cam_list = cam = processor = None

def init_camera():
    print("[Init] Setting up FLIR camera ...")

    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor
    
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
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        processor = PySpin.ImageProcessor()

        print("[Init] Camera Init OK")

    except Exception as e:
        CAM_ERROR_MSG = f"Camera init error: {e!r}"
        CAM_OK = False
        print(CAM_ERROR_MSG)
        return
    
def set_led_camera_params(led_id: int):
    if cam is None:
        return
    exp = LED_EXPOSURE_US.get(led_id, None)
    gain = LED_GAIN_DB.get(led_id, None)
    cam.ExposureTime.SetValue(exp)
    cam.Gain.SetValue(gain)

def is_door_closed(frame_gray: np.ndarray) -> tuple[bool, str]:
    x1, y1, x2, y2 = DOOR_ROI
    roi = frame_gray[y1:y2, x1:x2]

    try:
        aruco = cv2.aruco
    except AttributeError:
        return False, "OpenCV ArUco module not available. Install opencv-contrib-python."

    # Choose your dictionary (must match what you printed)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(roi)

    if ids is None or len(ids) == 0:
        print("No markers detected")
        return False, "Door marker not detected. Close the door fully."

    ids = ids.flatten().tolist()

    if DOOR_MARKER_ID is not None and DOOR_MARKER_ID not in ids:
        print("Wrong marker ID detected:", ids)
        return False, f"Wrong marker ID detected ({ids}). Expected {DOOR_MARKER_ID}."

    # Choose the marker we care about
    idx = ids.index(DOOR_MARKER_ID) if DOOR_MARKER_ID in ids else 0
    c = corners[idx].reshape(-1, 2)

    # Marker area filter (reject tiny / partial)
    area = cv2.contourArea(c.astype(np.float32))
    if area < DOOR_MIN_AREA:
        return False, "Door marker too small/partial. Ensure door is fully closed."

    return True, "Door closed"

def capture_image():
    global CAM_OK, CAM_ERROR_MSG
    x1, y1, x2, y2 = TRAY_ROI

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
    ok, reason = is_door_closed(arr)
    '''if not ok:
        return ("DOOR_OPEN", reason)'''
    return arr[y1:y2, x1:x2]

def flat_field_normalize(img: np.ndarray, led_id: int) -> np.ndarray:
    """
    Normalize an image using the latest calibration flat for this LED.
    N(x,y) = I(x,y) * (mean(flat) / flat(x,y))
    """
    cal = calibration_flats.get(led_id, None)
    if cal is None:
        #return original
        return img.copy()

    img = img.astype(np.float32)
    flat = cal.astype(np.float32)

    # Avoid divide-by-zero
    flat_safe = np.where(flat < 1.0, 1.0, flat)
    flat_mean = flat_safe.mean()

    norm = img * (flat_mean / flat_safe)
    norm = np.clip(norm, 0, 255).astype("uint8")
    return norm

def calib_flat_path_latest(led_id: int) -> str:
    return os.path.join(CALIB_DIR, f"LED{led_id}_flat_latest.npy")

def calib_flat_path_ref(led_id: int) -> str:
    return os.path.join(CALIB_DIR, f"LED{led_id}_flat_ref.npy")

def load_calibration_flats():
    global calibration_flats
    calibration_flats = {}
    for led_id in (1, 2, 3):
        path = calib_flat_path_latest(led_id)
        if os.path.exists(path):
            try:
                arr = np.load(path)
                calibration_flats[led_id] = arr
                print(f"[Calib] Loaded latest flat for LED {led_id} from {path}")
            except Exception as e:
                print(f"[Calib] Failed to load flat for LED {led_id}: {e}")

def save_calibration_flat(led_id: int, arr: np.ndarray, as_reference_if_missing=True):
    latest_path = calib_flat_path_latest(led_id)
    np.save(latest_path, arr)
    print(f"[Calib] Saved latest flat for LED {led_id} -> {latest_path}")

    ref_path = calib_flat_path_ref(led_id)
    if as_reference_if_missing and not os.path.exists(ref_path):
        np.save(ref_path, arr)
        print(f"[Calib] Saved reference flat for LED {led_id} -> {ref_path}")

def compare_to_reference(led_id: int, arr: np.ndarray) -> list[str]:
    """
    Compare current calibration arr with reference flat and produce warnings.
    Returns a list of warning strings (can be empty).
    """
    warnings = []
    ref_path = calib_flat_path_ref(led_id)
    if not os.path.exists(ref_path):
        return warnings

    try:
        ref = np.load(ref_path).astype(np.float32)
    except Exception as e:
        warnings.append(f"Could not load reference flat for LED {led_id}: {e}")
        return warnings

    cur = arr.astype(np.float32)

    ref_mean = ref.mean()
    cur_mean = cur.mean()
    if ref_mean < 1:
        return warnings

    ratio = cur_mean / ref_mean

    # Global brightness drift
    if abs(ratio - 1.0) > 0.1:   #10% difference
        warnings.append(
            f"LED {led_id}: Mean brightness drift {((ratio-1)*100):.1f}% "
            "relative to reference. LED output or exposure may have changed."
        )

    # Spatial deviation
    ref_safe = np.where(ref < 1.0, 1.0, ref)
    dev = np.abs(cur - ref_safe) / ref_safe
    mean_dev = dev.mean()
    max_dev = dev.max()

    if mean_dev > 0.1:
        warnings.append(
            f"LED {led_id}: Average spatial deviation {mean_dev*100:.1f}% "
            "from reference. Diffuser or LED array may have changed."
        )
    if max_dev > 0.25:
        warnings.append(
            f"LED {led_id}: Local deviation up to {max_dev*100:.1f}% from reference. "
            "Check for dirt, damage, or partial LED failure."
        )

    return warnings

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
        self.set_status("Initializing camera…")
        self.update_idletasks()
        init_camera()

        if not CAM_OK:
            messagebox.showerror(
                "Camera error",
                f"Could not initialize camera:\n{CAM_ERROR_MSG} - check the connection and use Settings to reconnect"
            )
            return

        load_calibration_flats()
        if calibration_flats:
            self.set_status("Camera initialized (calibration found).")
            self.start_btn.config(state="normal")
        else:
            self.set_status("Camera initialized (no calibration).")
            self.calibrate_camera()

    def calibrate_camera(self):
        global calibration_flats

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
            "Prepare White Board",
            "Remove peanut tray and insert the full-size white calibration board.\n"
            "Click OK to continue."
        )
        
        self.set_status("Capturing calibration flats…")
        self.update_idletasks()

        leds_to_calibrate = [(1, led1),(2, led2),(3, led3)]
        calibration_flats = {}
        all_warnings = []

        try:
            for led_id, led_dev in leds_to_calibrate:
                self.set_status(f"Calibrating LED {led_id}…")
                self.update_idletasks()

                set_led_camera_params(led_id)

                driver.on()
                led_dev.on()
                time.sleep(0.3)

                img = capture_image()

                '''if isinstance(img, tuple) and img[0] == "DOOR_OPEN":
                    reason = img[1]
                    self.after(0, lambda: messagebox.showwarning("Door Not Closed", reason))
                    return'''

                led_dev.off()
                driver.off()
                time.sleep(0.2)

                if img is None:
                    all_warnings.append(f"LED {led_id}: Failed to capture calibration image.")
                    continue

                calibration_flats[led_id] = img

                # Save latest flat and reference if missing
                save_calibration_flat(led_id, img, as_reference_if_missing=True)
                calib_raw_name = os.path.join(CALIB_DIR, f"LED{led_id}_CALIB_RAW.png")
                cv2.imwrite(calib_raw_name, img)

                # Compare to reference and collect warnings
                w = compare_to_reference(led_id, img)
                all_warnings.extend(w)

            if all_warnings:
                msg = "\n\n".join(all_warnings)
                messagebox.showwarning(
                    "Calibration Warnings",
                    msg + "\n\nCheck LEDs, diffuser, and calibration board."
                )

            self.set_status("Ready (calibrated)")
            self.start_btn.config(state="normal")
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
                "Camera not ready",
                "Camera not connected. Use Settings → Reconnect Camera."
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
            "Insert the peanut Tray with peanuts.\n"
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
            leds = [(1, led1), (2, led2), (3, led3)]
            total_steps = len(leds)

            for idx, (i, led) in enumerate(leds, start=1):
                self.safe_status(f"Capturing LED {i}...")
                set_led_camera_params(i)
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

                img_norm = flat_field_normalize(img, i)

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                raw_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED{i}_raw_.png")
                norm_name = os.path.join(IMAGE_DIR, f"{timestamp}_LED{i}_norm.png")

                cv2.imwrite(raw_name, img)         
                cv2.imwrite(norm_name, img_norm)

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
