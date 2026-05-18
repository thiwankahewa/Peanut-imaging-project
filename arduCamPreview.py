#!/usr/bin/env python3
import os
import time
import threading
import subprocess
import platform
import cv2
import tkinter as tk
from tkinter import ttk, messagebox

# ============================================================
# CONFIG
# ============================================================

DEVICE = "/dev/video0"

# ---------- GPIO / LED CONFIG ----------
# Same style as your uploaded code
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

if platform.system() == "Linux":
    from gpiozero import OutputDevice
else:
    # Mock for non-Linux testing
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            print("[MOCK] OutputDevice created")
        def on(self):
            print("[MOCK] ON")
        def off(self):
            print("[MOCK] OFF")
        def close(self):
            print("[MOCK] CLOSE")

# Relay pin definitions (BCM)
DRIVER_PIN = 17
LED1_PIN   = 27
LED2_PIN   = 22
LED3_PIN   = 23
LED4_PIN   = 24   # <-- THIS IS LED4

# Initialize relays
# Keep same pattern as your code
print("[Init] Setting up GPIO...")
driver = OutputDevice(DRIVER_PIN, active_high=True, initial_value=False)
led1   = OutputDevice(LED1_PIN,   active_high=True, initial_value=False)
led2   = OutputDevice(LED2_PIN,   active_high=True, initial_value=False)
led3   = OutputDevice(LED3_PIN,   active_high=True, initial_value=False)
led4   = OutputDevice(LED4_PIN,   active_high=True, initial_value=False)


def run_v4l2(args):
    """Run v4l2-ctl command safely."""
    cmd = ["v4l2-ctl", "-d", DEVICE] + args
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("v4l2-ctl error:", result.stderr.strip())
    except Exception as e:
        print(f"v4l2 error: {e}")


def cleanup_hardware():
    """Turn everything off and release GPIO."""
    print("[Cleanup] Releasing hardware...")
    try:
        led1.off()
        led2.off()
        led3.off()
        led4.off()
        driver.off()
    except Exception:
        pass

    for dev in [driver, led1, led2, led3, led4]:
        try:
            dev.close()
        except Exception:
            pass
    print("[Cleanup] Done.")


class CameraControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("USB Camera Preview + Controls + LED4")
        self.root.geometry("950x720")

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Optional preview resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.last_update_time = 0
        self.update_delay = 0.08  # throttle slider updates

        # LED states
        self.led1_on = False
        self.led2_on = False
        self.led3_on = False
        self.led4_on = False

        self.build_ui()
        self.apply_all()

        self.preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        self.preview_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ========================================================
    # UI
    # ========================================================
    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="Camera Controls", padding=10)
        controls.pack(fill="x", padx=5, pady=5)

        row = 0

        # -------- Auto exposure mode --------
        ttk.Label(controls, text="Auto Exposure").grid(row=row, column=0, sticky="w")
        self.auto_exposure_var = tk.StringVar(value="3")
        self.auto_exposure_combo = ttk.Combobox(
            controls,
            textvariable=self.auto_exposure_var,
            state="readonly",
            values=["0", "1", "2", "3"],
            width=12
        )
        self.auto_exposure_combo.grid(row=row, column=1, sticky="ew", padx=5)
        self.auto_exposure_combo.bind("<<ComboboxSelected>>", self.on_auto_exposure_change)
        ttk.Label(controls, text="0=Auto, 1=Manual, 2=Shutter, 3=Aperture").grid(row=row, column=2, sticky="w")
        row += 1

        # -------- Exposure slider --------
        self.exposure_var = tk.IntVar(value=157)
        self.add_slider(controls, "Exposure Time Absolute", self.exposure_var, 1, 5000, row, self.on_exposure_change)
        self.exposure_scale = controls.grid_slaves(row=row, column=1)[0]
        row += 1

        # -------- Auto white balance --------
        self.wb_auto_var = tk.IntVar(value=1)
        self.wb_check = ttk.Checkbutton(
            controls,
            text="White Balance Automatic",
            variable=self.wb_auto_var,
            command=self.on_wb_auto_change
        )
        self.wb_check.grid(row=row, column=0, sticky="w")
        row += 1

        # -------- White balance temperature --------
        self.wb_temp_var = tk.IntVar(value=4600)
        self.add_slider(controls, "White Balance Temperature", self.wb_temp_var, 2800, 6500, row, self.on_wb_temp_change)
        self.wb_temp_scale = controls.grid_slaves(row=row, column=1)[0]
        row += 1

        # -------- Other sliders --------
        self.brightness_var = tk.IntVar(value=0)
        self.add_slider(controls, "Brightness", self.brightness_var, -64, 64, row, lambda v: self.set_control("brightness", v))
        row += 1

        self.contrast_var = tk.IntVar(value=32)
        self.add_slider(controls, "Contrast", self.contrast_var, 0, 64, row, lambda v: self.set_control("contrast", v))
        row += 1

        self.saturation_var = tk.IntVar(value=64)
        self.add_slider(controls, "Saturation", self.saturation_var, 0, 128, row, lambda v: self.set_control("saturation", v))
        row += 1

        self.hue_var = tk.IntVar(value=0)
        self.add_slider(controls, "Hue", self.hue_var, -40, 40, row, lambda v: self.set_control("hue", v))
        row += 1

        self.gamma_var = tk.IntVar(value=100)
        self.add_slider(controls, "Gamma", self.gamma_var, 72, 500, row, lambda v: self.set_control("gamma", v))
        row += 1

        self.gain_var = tk.IntVar(value=0)
        self.add_slider(controls, "Gain", self.gain_var, 0, 100, row, lambda v: self.set_control("gain", v))
        row += 1

        self.sharpness_var = tk.IntVar(value=3)
        self.add_slider(controls, "Sharpness", self.sharpness_var, 0, 6, row, lambda v: self.set_control("sharpness", v))
        row += 1

        self.backlight_var = tk.IntVar(value=1)
        self.add_slider(controls, "Backlight Compensation", self.backlight_var, 0, 2, row, lambda v: self.set_control("backlight_compensation", v))
        row += 1

        # -------- Power line frequency --------
        ttk.Label(controls, text="Power Line Frequency").grid(row=row, column=0, sticky="w")
        self.plf_var = tk.StringVar(value="2")
        self.plf_combo = ttk.Combobox(
            controls,
            textvariable=self.plf_var,
            state="readonly",
            values=["0", "1", "2"],
            width=12
        )
        self.plf_combo.grid(row=row, column=1, sticky="ew", padx=5)
        self.plf_combo.bind("<<ComboboxSelected>>", self.on_plf_change)
        ttk.Label(controls, text="0=Disabled, 1=50 Hz, 2=60 Hz").grid(row=row, column=2, sticky="w")
        row += 1

        for i in range(3):
            controls.columnconfigure(i, weight=1)

        # ========================================================
        # LED SECTION
        # ========================================================
        led_frame = ttk.LabelFrame(main, text="LED Control", padding=10)
        led_frame.pack(fill="x", padx=5, pady=10)

        ttk.Label(
            led_frame,
            text="LED4 is connected to BCM pin 24"
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        self.led1_btn = tk.Button(
            led_frame,
            text="LED 1",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            width=12,
            command=lambda: self.toggle_led_exclusive(led1, "led1_on", self.led1_btn, "LED 1")
        )
        self.led1_btn.grid(row=1, column=0, padx=5, pady=5)

        self.led2_btn = tk.Button(
            led_frame,
            text="LED 2",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            width=12,
            command=lambda: self.toggle_led_exclusive(led2, "led2_on", self.led2_btn, "LED 2")
        )
        self.led2_btn.grid(row=1, column=1, padx=5, pady=5)

        self.led3_btn = tk.Button(
            led_frame,
            text="LED 3",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            width=12,
            command=lambda: self.toggle_led_exclusive(led3, "led3_on", self.led3_btn, "LED 3")
        )
        self.led3_btn.grid(row=1, column=2, padx=5, pady=5)

        # --------------------------------------------------------
        # THIS IS THE LED4 BUTTON
        # When pressed, it turns ON driver + LED4
        # --------------------------------------------------------
        self.led4_btn = tk.Button(
            led_frame,
            text="LED 4",
            font=("Helvetica", 12),
            relief="raised",
            bd=2,
            width=12,
            command=lambda: self.toggle_led_exclusive(led4, "led4_on", self.led4_btn, "LED 4")
        )
        self.led4_btn.grid(row=1, column=3, padx=5, pady=5)

        self.led_default_bg = self.led1_btn.cget("bg")
        self.led_default_fg = self.led1_btn.cget("fg")

        quick_frame = ttk.Frame(led_frame)
        quick_frame.grid(row=2, column=0, columnspan=4, pady=(10, 0), sticky="w")

        ttk.Button(quick_frame, text="All LEDs OFF", command=self.turn_off_all_leds).pack(side="left", padx=5)

        # --------------------------------------------------------
        # DIRECT LED4 ON / OFF BUTTONS
        # These are the exact extra controls for LED4 only
        # --------------------------------------------------------
        ttk.Button(quick_frame, text="LED4 ON", command=self.turn_on_led4_direct).pack(side="left", padx=5)
        ttk.Button(quick_frame, text="LED4 OFF", command=self.turn_off_led4_direct).pack(side="left", padx=5)

        # ========================================================
        # ACTION BUTTONS
        # ========================================================
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(btn_frame, text="Apply All Camera Settings", command=self.apply_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Print Current Settings", command=self.print_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.on_close).pack(side="left", padx=5)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(5, 0))

    def add_slider(self, parent, label, variable, minv, maxv, row, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = tk.Scale(
            parent,
            from_=minv,
            to=maxv,
            orient="horizontal",
            variable=variable,
            command=command,
            resolution=1,
            length=420
        )
        scale.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(parent, textvariable=variable, width=8).grid(row=row, column=2, sticky="w")

    # ========================================================
    # Camera controls
    # ========================================================
    def set_status(self, text):
        self.status_var.set(text)

    def throttled(self, func, value):
        now = time.time()
        if now - self.last_update_time >= self.update_delay:
            self.last_update_time = now
            func(value)

    def set_control(self, name, value):
        self.throttled(lambda v: run_v4l2(["-c", f"{name}={int(float(v))}"]), value)

    def on_auto_exposure_change(self, event=None):
        mode = int(self.auto_exposure_var.get())
        run_v4l2(["-c", f"auto_exposure={mode}"])
        time.sleep(0.05)
        self.update_control_states()
        if mode == 1:
            self.on_exposure_change(self.exposure_var.get())

    def on_exposure_change(self, value):
        if int(self.auto_exposure_var.get()) == 1:
            self.throttled(
                lambda v: run_v4l2(["-c", f"exposure_time_absolute={int(float(v))}"]),
                value
            )

    def on_wb_auto_change(self):
        val = int(self.wb_auto_var.get())
        run_v4l2(["-c", f"white_balance_automatic={val}"])
        time.sleep(0.05)
        self.update_control_states()
        if val == 0:
            self.on_wb_temp_change(self.wb_temp_var.get())

    def on_wb_temp_change(self, value):
        if int(self.wb_auto_var.get()) == 0:
            self.throttled(
                lambda v: run_v4l2(["-c", f"white_balance_temperature={int(float(v))}"]),
                value
            )

    def on_plf_change(self, event=None):
        val = int(self.plf_var.get())
        run_v4l2(["-c", f"power_line_frequency={val}"])

    def update_control_states(self):
        manual_exposure = int(self.auto_exposure_var.get()) == 1
        self.exposure_scale.config(state="normal" if manual_exposure else "disabled")

        manual_wb = int(self.wb_auto_var.get()) == 0
        self.wb_temp_scale.config(state="normal" if manual_wb else "disabled")

    def apply_all(self):
        run_v4l2(["-c", f"auto_exposure={int(self.auto_exposure_var.get())}"])
        run_v4l2(["-c", f"white_balance_automatic={int(self.wb_auto_var.get())}"])
        run_v4l2(["-c", f"power_line_frequency={int(self.plf_var.get())}"])

        run_v4l2(["-c", f"brightness={self.brightness_var.get()}"])
        run_v4l2(["-c", f"contrast={self.contrast_var.get()}"])
        run_v4l2(["-c", f"saturation={self.saturation_var.get()}"])
        run_v4l2(["-c", f"hue={self.hue_var.get()}"])
        run_v4l2(["-c", f"gamma={self.gamma_var.get()}"])
        run_v4l2(["-c", f"gain={self.gain_var.get()}"])
        run_v4l2(["-c", f"sharpness={self.sharpness_var.get()}"])
        run_v4l2(["-c", f"backlight_compensation={self.backlight_var.get()}"])

        if int(self.auto_exposure_var.get()) == 1:
            run_v4l2(["-c", f"exposure_time_absolute={self.exposure_var.get()}"])

        if int(self.wb_auto_var.get()) == 0:
            run_v4l2(["-c", f"white_balance_temperature={self.wb_temp_var.get()}"])

        self.update_control_states()
        self.set_status("Camera settings applied")

    def print_settings(self):
        print("Current UI settings:")
        print(f"auto_exposure={self.auto_exposure_var.get()}")
        print(f"exposure_time_absolute={self.exposure_var.get()}")
        print(f"white_balance_automatic={self.wb_auto_var.get()}")
        print(f"white_balance_temperature={self.wb_temp_var.get()}")
        print(f"brightness={self.brightness_var.get()}")
        print(f"contrast={self.contrast_var.get()}")
        print(f"saturation={self.saturation_var.get()}")
        print(f"hue={self.hue_var.get()}")
        print(f"gamma={self.gamma_var.get()}")
        print(f"gain={self.gain_var.get()}")
        print(f"sharpness={self.sharpness_var.get()}")
        print(f"backlight_compensation={self.backlight_var.get()}")
        print(f"power_line_frequency={self.plf_var.get()}")
        print(f"LED4 state={self.led4_on}")

    # ========================================================
    # LED controls
    # ========================================================
    def update_led_button(self, btn, label, is_on):
        if is_on:
            btn.config(bg="#4caf50", fg="white", text=f"{label} (ON)")
        else:
            btn.config(bg=self.led_default_bg, fg=self.led_default_fg, text=label)

    def turn_off_all_leds(self):
        try:
            led1.off()
            led2.off()
            led3.off()
            led4.off()
            driver.off()
        except Exception as e:
            print("LED off error:", e)

        self.led1_on = False
        self.led2_on = False
        self.led3_on = False
        self.led4_on = False

        self.update_led_button(self.led1_btn, "LED 1", False)
        self.update_led_button(self.led2_btn, "LED 2", False)
        self.update_led_button(self.led3_btn, "LED 3", False)
        self.update_led_button(self.led4_btn, "LED 4", False)

        self.set_status("All LEDs OFF")

    def toggle_led_exclusive(self, target_led, state_attr_name, target_btn, label):
        current = getattr(self, state_attr_name)

        if current:
            self.turn_off_all_leds()
            return

        self.turn_off_all_leds()

        try:
            # ----------------------------------------------------
            # THESE ARE THE MAIN LINES THAT TURN THE SELECTED LED ON
            # For LED4, target_led = led4
            # ----------------------------------------------------
            driver.on()
            target_led.on()

            setattr(self, state_attr_name, True)
            self.update_led_button(target_btn, label, True)
            self.set_status(f"{label} ON")
        except Exception as e:
            self.set_status(f"{label} error: {e}")
            messagebox.showerror("LED Error", f"Failed to turn on {label}: {e}")

    # --------------------------------------------------------
    # EXACT DIRECT FUNCTIONS FOR LED4
    # These are the clearest lines if you want only LED4 control
    # --------------------------------------------------------
    def turn_on_led4_direct(self):
        try:
            self.turn_off_all_leds()
            driver.on()   # turn relay driver ON
            led4.on()     # turn LED4 ON
            self.led4_on = True
            self.update_led_button(self.led4_btn, "LED 4", True)
            self.set_status("LED 4 ON")
        except Exception as e:
            self.set_status(f"LED4 error: {e}")
            messagebox.showerror("LED4 Error", str(e))

    def turn_off_led4_direct(self):
        try:
            led4.off()    # turn LED4 OFF
            driver.off()  # optional: turn driver OFF too
            self.led4_on = False
            self.update_led_button(self.led4_btn, "LED 4", False)
            self.set_status("LED 4 OFF")
        except Exception as e:
            self.set_status(f"LED4 error: {e}")
            messagebox.showerror("LED4 Error", str(e))

    # ========================================================
    # Preview loop
    # ========================================================
    def preview_loop(self):
        print("Press q in preview window to quit")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            cv2.imshow("USB Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.on_close()
                break

        cv2.destroyAllWindows()

    # ========================================================
    # Close
    # ========================================================
    def on_close(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        cleanup_hardware()

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraControlApp(root)
    root.mainloop()