#!/usr/bin/env python3
import os
import time
import cv2
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

# Disable auto exposure and gain
nodemap = cam.GetNodeMap()
exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
exp_auto_off = exp_auto.GetEntryByName("Off")
exp_auto.SetIntValue(exp_auto_off.GetValue())

gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
gain_auto_off = gain_auto.GetEntryByName("Off")
gain_auto.SetIntValue(gain_auto_off.GetValue())

# Manual exposure/gain settings
exp_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
exp_time.SetValue(5000)  # microseconds
gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
gain.SetValue(0.0)

# --- Capture directory ---
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# --- Helper: capture one image ---
def capture_image(name):
    cam.BeginAcquisition()
    processor = PySpin.ImageProcessor()
    img = cam.GetNextImage(1000)
    if not img.IsIncomplete():
        arr = processor.Convert(img, PySpin.PixelFormat_Mono8).GetNDArray()
        cv2.imwrite(name, arr)
        print(f"[✓] Saved {name}")
    else:
        print("[x] Incomplete image.")
    img.Release()
    cam.EndAcquisition()

# --- Main LED sequence ---
try:
    leds = [(1, led1), (2, led2), (3, led3)]
    for i, led in leds:
        print(f"--> Capturing LED {i}")
        driver.on()
        time.sleep(0.2)

        led.on()
        time.sleep(1)  # stabilize lighting

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"image_LED{i}_{timestamp}.png")
        capture_image(filename)

        led.off()
        driver.off()
        time.sleep(0.3)

    print("All captures complete!")

except KeyboardInterrupt:
    print("\n[!] Interrupted by user.")
finally:
    # Turn everything off and release
    for dev in [driver, led1, led2, led3]:
        dev.off()
        dev.close()

    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    print("GPIO and camera released.")
