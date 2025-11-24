#!/usr/bin/env python3
import os
import time
import tkinter as tk
# We no longer import scrolledtext
import threading

# --- Development Flag ---
USE_MOCK_HARDWARE = False

# --- Mock Hardware Classes (for development) ---
class MockOutputDevice:
    """Mocks the gpiozero.OutputDevice for development without hardware."""
    def __init__(self, pin, active_high=False, initial_value=False):
        self.pin = pin
        self.active_high = active_high
        self.value = initial_value
        self.log_func = print
        self.log_func(f"[MockGPIO] Initialized pin {self.pin}")
    
    def on(self):
        self.value = True
        self.log_func(f"[MockGPIO Pin {self.pin}] -> ON")
        
    def off(self):
        self.value = False
        self.log_func(f"[MockGPIO Pin {self.pin}] -> OFF")
        
    def close(self):
        self.value = False
        self.log_func(f"[MockGPIO Pin {self.pin}] -> Closed")

# --- Import real or mock hardware based on the flag ---
if USE_MOCK_HARDWARE:
    OutputDevice = MockOutputDevice
else:
    os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"
    from gpiozero import OutputDevice
    import PySpin
    import cv2
    
# --- Relay pin definitions (BCM) ---
DRIVER = 17
LED1   = 27
LED2   = 22
LED3   = 23

# --- Initialize relays (will use Mock or Real based on flag) ---
driver = OutputDevice(DRIVER, active_high=False, initial_value=False)
led1   = OutputDevice(LED1,   active_high=False, initial_value=False)
led2   = OutputDevice(LED2,   active_high=False, initial_value=False)
led3   = OutputDevice(LED3,   active_high=False, initial_value=False)

# --- Camera setup (commented out as in original) ---
'''
system = PySpin.System.GetInstance()
# ... (all your original camera setup code) ...
'''

# --- Helper: capture one image ---
def capture_image(name, log_func):
    if USE_MOCK_HARDWARE:
        log_func(f"Simulating capture for {name}...")
        time.sleep(0.5)
        log_func(f"[✓] (Simulated) Saved {name}")
    else:
        # --- Paste your REAL capture_image logic here ---
        log_func(f"Attempting REAL capture for {name}...")
        pass

# --- Main LED sequence (Turned into a function) ---
def run_capture_sequence(log_func):
    if USE_MOCK_HARDWARE:
        for dev in [driver, led1, led2, led3]:
            dev.log_func = log_func

    try:
        leds = [(1, led1), (2, led2), (3, led3)]
        for i, led in leds:
            log_func(f"--> Capturing LED {i}")
            driver.on()
            time.sleep(0.2)

            led.on()
            time.sleep(1)

            filename = f"image_LED{i}.png"
            capture_image(filename, log_func)

            led.off()
            driver.off()
            time.sleep(0.3)

        log_func("All captures complete!")

    except KeyboardInterrupt:
        log_func("\n[!] Interrupted by user.")
    except Exception as e:
        log_func(f"\n[ERROR] An error occurred: {e}")
    finally:
        log_func("Cleaning up GPIO...")
        for dev in [driver, led1, led2, led3]:
            dev.off()
        log_func("Sequence finished.")

# -----------------------------------------------------------------
# ---                 Tkinter GUI Application                   ---
# -----------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hardware Capture GUI")
        self.root.geometry("500x450")

        # Main frame (using tk.Frame)
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls (using tk.LabelFrame)
        control_frame = tk.LabelFrame(main_frame, text="Controls", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=5)

        # --- Create GUI Widgets ---
        self.create_manual_controls(control_frame)
        
        # Separator (using a simple Frame)
        tk.Frame(control_frame, height=2, bd=1, relief='sunken').pack(fill='x', pady=10)

        # Run button (using tk.Button)
        self.run_button = tk.Button(control_frame, text="Run Full Capture Sequence", command=self.start_sequence_thread)
        self.run_button.pack(fill=tk.X)

        # Log (using tk.LabelFrame)
        log_frame = tk.LabelFrame(main_frame, text="Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- NEW: Create Scrollbar and Text widget manually ---
        self.scrollbar = tk.Scrollbar(log_frame, orient='vertical')
        self.log_widget = tk.Text(log_frame, 
                                  wrap=tk.WORD, 
                                  height=10, 
                                  state="disabled", 
                                  bg="black", 
                                  fg="white",
                                  yscrollcommand=self.scrollbar.set)
        
        # Configure the scrollbar to work with the text widget
        self.scrollbar.config(command=self.log_widget.yview)
        
        # Pack them to fill the log_frame
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # --- END OF NEW SECTION ---
        
        if USE_MOCK_HARDWARE:
            for dev in [driver, led1, led2, led3]:
                dev.log_func = self.log

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_manual_controls(self, parent):
        """Creates the manual toggle checkboxes."""
        manual_frame = tk.Frame(parent)
        manual_frame.pack(fill=tk.X)

        tk.Label(manual_frame, text="Manual Toggles:", font='-weight bold').pack(anchor='w')

        self.driver_var = tk.BooleanVar()
        self.led1_var = tk.BooleanVar()
        self.led2_var = tk.BooleanVar()
        self.led3_var = tk.BooleanVar()

        tk.Checkbutton(manual_frame, text="Driver", variable=self.driver_var, command=self.toggle_driver).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(manual_frame, text="LED 1", variable=self.led1_var, command=self.toggle_led1).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(manual_frame, text="LED 2", variable=self.led2_var, command=self.toggle_led2).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(manual_frame, text="LED 3", variable=self.led3_var, command=self.toggle_led3).pack(side=tk.LEFT, padx=5)

    def log(self, message):
        self.root.after(0, self._log_update, message)

    def _log_update(self, message):
        self.log_widget.config(state="normal")
        self.log_widget.insert(tk.END, f"{message}\n")
        self.log_widget.see(tk.END)
        self.log_widget.config(state="disabled")

    # --- Manual Toggle Callbacks ---
    def toggle_driver(self):
        if self.driver_var.get():
            driver.on()
        else:
            driver.off()
            
    def toggle_led1(self):
        if self.led1_var.get():
            led1.on()
        else:
            led1.off()

    def toggle_led2(self):
        if self.led2_var.get():
            led2.on()
        else:
            led2.off()

    def toggle_led3(self):
        if self.led3_var.get():
            led3.on()
        else:
            led3.off()

    # --- Sequence Threading ---
    def start_sequence_thread(self):
        self.run_button.config(state="disabled")
        self.log("Disabling manual toggles for sequence...")
        self.driver_var.set(False)
        self.led1_var.set(False)
        self.led2_var.set(False)
        self.led3_var.set(False)
        driver.off()
        led1.off()
        led2.off()
        led3.off()
        
        self.log("--- Starting sequence thread ---")
        self.sequence_thread = threading.Thread(
            target=self.run_sequence_wrapper,
            daemon=True
        )
        self.sequence_thread.start()

    def run_sequence_wrapper(self):
        try:
            run_capture_sequence(self.log)
        except Exception as e:
            self.log(f"[FATAL ERROR] {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state="normal"))

    def on_closing(self):
        self.log("Closing application and cleaning up...")
        for dev in [driver, led1, led2, led3]:
            dev.off()
            dev.close()
        self.log("All devices closed.")
        
        # if not USE_MOCK_HARDWARE:
        #     ... (real camera cleanup) ...
            
        self.root.destroy()

# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
