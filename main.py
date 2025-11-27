#!/usr/bin/env python3
import os
import platform
import time
import threading

import cv2
import PySpin
from gpiozero import OutputDevice

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# ============================================================
#  CONFIG
# ============================================================

# Directory to save images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

if "microsoft" in platform.release().lower():
    os.environ["GPIOZERO_PIN_FACTORY"] = "mock"
else:
    os.environ["GPIOZERO_PIN_FACTORY"] = "rpigpio"   # Force a working GPIO backend (must be set before creating OutputDevice)

# Relay pin definitions (BCM)
DRIVER_PIN = 17
LED1_PIN   = 27
LED2_PIN   = 22
LED3_PIN   = 23

# Exposure / gain settings
EXPOSURE_US = 12500.0  # microseconds
GAIN_DB     = 33.0


# ============================================================
#  HARDWARE INIT
# ============================================================

print("[Init] Setting up GPIO...")
driver = OutputDevice(DRIVER_PIN, active_high=False, initial_value=False)
led1   = OutputDevice(LED1_PIN,   active_high=False, initial_value=False)
led2   = OutputDevice(LED2_PIN,   active_high=False, initial_value=False)
led3   = OutputDevice(LED3_PIN,   active_high=False, initial_value=False)

print("[Init] Setting up FLIR camera (PySpin)...")
CAM_OK = False
CAM_ERROR_MSG = ""
system = None
cam_list = None
cam = None
processor = None

def init_camera():
    """
    Initialize the FLIR camera safely.
    Sets CAM_OK / CAM_ERROR_MSG.
    Only treats 'no camera' as a hard failure.
    """
    global CAM_OK, CAM_ERROR_MSG, system, cam_list, cam, processor

    print("[Init] Setting up FLIR camera (PySpin)...")
    CAM_OK = False
    CAM_ERROR_MSG = ""
    system = cam_list = cam = processor = None

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()

        num_cams = cam_list.GetSize()
        print(f"[Init] Number of cameras detected: {num_cams}")

        if num_cams == 0:
            CAM_ERROR_MSG = "No FLIR camera found"
            print("[Init] No cameras detected.")
            return  # leave CAM_OK = False

        cam = cam_list.GetByIndex(0)
        print("[Init] Got camera at index 0, initializing...")
        cam.Init()

        nodemap = cam.GetNodeMap()
        pf = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))

        mono16_entry = pf.GetEntryByName("Mono16")
        if PySpin.IsAvailable(mono16_entry) and PySpin.IsReadable(mono16_entry):
            pf.SetIntValue(mono16_entry.GetValue())
            current = PySpin.CEnumEntryPtr(pf.GetCurrentEntry())
            print("PixelFormat now:", current.GetSymbolic())
        else:
            print("Mono16 not available")

        # At this point, camera is usable
        CAM_OK = True
        processor = PySpin.ImageProcessor()
        print("[Init] Camera Init OK, now configuring exposure/gain...")

    except Exception as e:
        CAM_ERROR_MSG = f"Camera init error: {e!r}"
        CAM_OK = False
        print(f"[Init] Camera init failed with exception: {CAM_ERROR_MSG}")
        return

    try:
        nodemap = cam.GetNodeMap()

        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
            exp_auto_off = exp_auto.GetEntryByName("Off")
            exp_auto.SetIntValue(exp_auto_off.GetValue())
        else:
            print("[Init] Warning: ExposureAuto node not available/writable")

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
            gain_auto_off = gain_auto.GetEntryByName("Off")
            gain_auto.SetIntValue(gain_auto_off.GetValue())
        else:
            print("[Init] Warning: GainAuto node not available/writable")

        exp_time_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exp_time_node) and PySpin.IsWritable(exp_time_node):
            exp_time_node.SetValue(EXPOSURE_US)
        else:
            print("[Init] Warning: ExposureTime node not available/writable")

        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
            gain_node.SetValue(GAIN_DB)
        else:
            print("[Init] Warning: Gain node not available/writable")

        print("[Init] Camera configuration done.")

    except Exception as e:
        # Configuration failed, but we still consider the camera usable
        print(f"[Init] Warning: failed to configure camera nodes: {e!r}")



def capture_image(filepath: str):
    if not CAM_OK or cam is None or processor is None:
        raise RuntimeError("Camera not initialized")

    cam.BeginAcquisition()
    img = cam.GetNextImage(1000)

    if not img.IsIncomplete():
        print(img.GetPixelFormat())
        print("Source pixel format:", img.GetPixelFormatName()) 
        arr = processor.Convert(img, PySpin.PixelFormat_Mono16).GetNDArray()
        arr = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(f"[Debug] pixel min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
        cv2.imwrite(filepath, arr)
        print(f"[✓] Saved {filepath}")
    else:
        print("[x] Incomplete image.")

    img.Release()
    cam.EndAcquisition()


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

        if "microsoft" in platform.release().lower():
            self.geometry("800x480")
            self.resizable(False, False)
        else:
            self.attributes("-fullscreen",True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.capture_thread = None
        self.is_capturing = False
        self.preview_img = None  # keep reference to avoid GC

        # Tk variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var   = tk.StringVar(value="Idle")

        self._create_style()
        self._create_widgets()

    # ---------------- Styles ----------------
    def _create_style(self):
        style = ttk.Style(self)
        style.configure("TButton", font=("Helvetica", 14))
        style.configure("TLabel",  font=("Helvetica", 12))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))

    # ---------------- Layout ----------------
    def _create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # ==== Capture tab ====
        self.tab_capture = ttk.Frame(notebook)
        notebook.add(self.tab_capture, text="Capture")

        self.tab_capture.columnconfigure(0, weight=1)
        for r in range(4):
            self.tab_capture.rowconfigure(r, weight=1)

        header_lbl = ttk.Label(
            self.tab_capture,
            text="Peanut Imaging Sequence",
            style="Header.TLabel"
        )
        header_lbl.grid(row=0, column=0, pady=(20, 10), sticky="n")

        self.start_btn = ttk.Button(
            self.tab_capture,
            text="Start Capture",
            command=self.on_start_capture
        )
        self.start_btn.grid(row=1, column=0, pady=10, ipadx=40, ipady=15)

        self.progress_bar = ttk.Progressbar(
            self.tab_capture,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=2, column=0, padx=40, pady=10, sticky="ew")

        self.status_label = ttk.Label(
            self.tab_capture,
            textvariable=self.status_var
        )
        self.status_label.grid(row=3, column=0, pady=(0, 20))

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

        header = ttk.Label(
            self.tab_settings,
            text="Settings & Manual Control",
            style="Header.TLabel"
        )
        header.grid(row=0, column=0, pady=(20, 10))

        # ---- Manual LED test buttons ----
        leds_frame = ttk.LabelFrame(self.tab_settings, text="Manual LED Test")
        leds_frame.grid(row=1, column=0, pady=10, padx=40, sticky="ew")

        btn_led1 = ttk.Button(
            leds_frame, text="Pulse LED 1",
            command=lambda: self.test_led(led1)
        )
        btn_led1.grid(row=0, column=0, padx=5, pady=5)

        btn_led2 = ttk.Button(
            leds_frame, text="Pulse LED 2",
            command=lambda: self.test_led(led2)
        )
        btn_led2.grid(row=0, column=1, padx=5, pady=5)

        btn_led3 = ttk.Button(
            leds_frame, text="Pulse LED 3",
            command=lambda: self.test_led(led3)
        )
        btn_led3.grid(row=0, column=2, padx=5, pady=5)

        # ---- Camera controls ----
        cam_frame = ttk.LabelFrame(self.tab_settings, text="Camera")
        cam_frame.grid(row=2, column=0, pady=10, padx=40, sticky="ew")

        reconnect_btn = ttk.Button(
            cam_frame, text="Reconnect Camera",
            command=self.on_reconnect_camera
        )
        reconnect_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # ---- Exit button ----
        exit_btn = ttk.Button(
            self.tab_settings,
            text="Exit to Desktop",
            command=self.on_close
        )
        exit_btn.grid(row=3, column=0, pady=(10, 30), ipadx=20, ipady=5)

        if not CAM_OK:
            self.set_status("Camera not detected - use Settings or Start to retry")
            # Show popup error
            self.after(
                100,
                lambda: messagebox.showerror(
                    "Camera error",
                    f"No FLIR camera detected:\n{CAM_ERROR_MSG}"
                )
            )
        else:
            self.set_status("Ready")

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
        """Thread-safe gallery reload."""
        self.after(0, self.load_image_list)

    def test_led(self, led_dev):
        """Briefly turn on one LED for testing, if not capturing."""
        if self.is_capturing:
            messagebox.showinfo(
                "Busy",
                "Cannot test LEDs while capture is running."
            )
            return

        try:
            self.set_status("Testing LED...")
            driver.on()
            led_dev.on()
            self.update_idletasks()
            # short pulse without blocking the whole GUI for long
            self.after(200, lambda: self._finish_led_test(led_dev))
        except Exception as e:
            messagebox.showerror("LED error", f"Failed to test LED: {e}")
            self.set_status(f"Error testing LED: {e}")

    def _finish_led_test(self, led_dev):
        """Turn LED + driver off after pulse."""
        try:
            led_dev.off()
            driver.off()
        except Exception:
            pass
        self.set_status("Ready")

    def on_reconnect_camera(self):
        """Try to reinitialize the camera from Settings tab."""
        global CAM_OK, CAM_ERROR_MSG

        if self.is_capturing:
            messagebox.showinfo(
                "Busy",
                "Cannot reconnect camera while capture is running."
            )
            return

        self.set_status("Reconnecting camera...")
        self.update_idletasks()

        init_camera()  # updates CAM_OK and CAM_ERROR_MSG

        if CAM_OK:
            self.set_status("Camera connected (Ready)")
            messagebox.showinfo(
                "Camera",
                "Camera reconnected successfully."
            )
        else:
            self.set_status("Camera not available")
            messagebox.showerror(
                "Camera error",
                f"Failed to connect to camera:\n{CAM_ERROR_MSG}"
        )


    # =======================================================
    #  Capture Flow
    # =======================================================

    def on_start_capture(self):
        global CAM_OK, CAM_ERROR_MSG
        """Start capture sequence when button is pressed."""
        if self.is_capturing:
            return
        
        if not CAM_OK:
            self.set_status("Trying to connect to camera...")
            self.update_idletasks()

            init_camera()  # this updates CAM_OK and CAM_ERROR_MSG

            if not CAM_OK:
                # Still no camera after retry
                self.set_status("Camera not available")
                messagebox.showerror(
                    "Camera error",
                    f"Camera not detected:\n{CAM_ERROR_MSG}"
                )
                return
            else:
                self.set_status("Camera connected. Starting capture...")

        self.is_capturing = True
        self.start_btn.config(state="disabled")
        self.set_status("Starting capture...")
        self.set_progress(0.0)

        self.capture_thread = threading.Thread(target=self.capture_sequence)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_sequence(self):
        """
        3-LED capture sequence (runs in background thread).
        Uses global driver/LEDs + capture_image() function.
        """
        if not CAM_OK:
            self.safe_status("Camera not available")
            self.safe_capture_end()
            return
    
        try:
            leds = [(1, led1), (2, led2), (3, led3)]
            total_steps = len(leds)

            for idx, (i, led) in enumerate(leds, start=1):
                self.safe_status(f"Capturing LED {i}...")

                driver.on()
                time.sleep(0.2)

                led.on()
                time.sleep(1.0)  # let light stabilize

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{i}_{timestamp}_{i}.png"
                filepath = os.path.join(IMAGE_DIR, filename)

                capture_image(filepath)

                led.off()
                driver.off()
                time.sleep(0.3)

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
                import numpy as np
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
        """Handle window close; make sure hardware is cleaned up."""
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
        init_camera()
        app = PeanutApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        cleanup_hardware()
