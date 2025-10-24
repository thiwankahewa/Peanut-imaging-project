#!/usr/bin/env python3
import time
import argparse
import pathlib
import datetime
import sys

# GPIO
from gpiozero import OutputDevice

# Camera (PySpin)
import PySpin

# We'll use OpenCV just to save PNGs from numpy arrays
import cv2

# --------------- GPIO WRAPPER ----------------
class Relay:
    """
    Simple relay wrapper using gpiozero.OutputDevice.
    If your relay is active-LOW, set active_low=True (default here).
    """
    def __init__(self, pin_bcm: int, active_low: bool = True, name: str = ""):
        self.name = name or f"GPIO{pin_bcm}"
        # For active-low relays: active_high=False means .on() drives pin LOW.
        self.dev = OutputDevice(pin_bcm, active_high=not active_low, initial_value=False)

    def on(self):
        self.dev.on()

    def off(self):
        self.dev.off()

    def close(self):
        self.dev.close()

# --------------- PYSPIN HELPERS ----------------
def init_pyspin_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR/Spinnaker camera detected.")
    cam = cam_list.GetByIndex(0)
    cam.Init()

    # Single frame mode
    nodemap = cam.GetNodeMap()
    acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    acq_single = acq_mode.GetEntryByName("SingleFrame")
    acq_mode.SetIntValue(acq_single.GetValue())

    return system, cam, cam_list

def set_manual_exposure_gain(cam, exposure_us=None, gain_db=None):
    """Disable auto exposure/gain and set manual values."""
    nm = cam.GetNodeMap()

    # ExposureAuto OFF
    expo_auto = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
    expo_off = expo_auto.GetEntryByName("Off")
    expo_auto.SetIntValue(expo_off.GetValue())

    # GainAuto OFF
    gain_auto = PySpin.CEnumerationPtr(nm.GetNode("GainAuto"))
    gain_off = gain_auto.GetEntryByName("Off")
    gain_auto.SetIntValue(gain_off.GetValue())

    if exposure_us is not None:
        expo = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
        # Clamp to camera range just in case
        exposure_us = max(min(exposure_us, expo.GetMax()), expo.GetMin())
        expo.SetValue(exposure_us)

    if gain_db is not None:
        gain = PySpin.CFloatPtr(nm.GetNode("Gain"))
        gain_db = max(min(gain_db, gain.GetMax()), gain.GetMin())
        gain.SetValue(gain_db)

def grab_pyspin_image(cam, timeout_ms=2000):
    cam.BeginAcquisition()
    img = cam.GetNextImage(timeout_ms)
    if img.IsIncomplete():
        st = img.GetImageStatus()
        img.Release()
        cam.EndAcquisition()
        raise RuntimeError(f"Incomplete image (status {st})")
    # Convert to Mono8 numpy array
    converted = img.Convert(PySpin.PixelFormat_Mono8, PySpin.DIRECTIONAL_FILTER)
    arr = converted.GetNDArray()
    img.Release()
    cam.EndAcquisition()
    return arr

def release_pyspin(system, cam, cam_list):
    try:
        cam.DeInit()
    except Exception:
        pass
    cam_list.Clear()
    system.ReleaseInstance()

# --------------- SEQUENCE LOGIC ----------------
def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi GPIO relays + FLIR capture per LED")
    ap.add_argument("--driver-pin", type=int, default=17, help="BCM pin for driver/DIM relay")
    ap.add_argument("--led1-pin", type=int, default=27, help="BCM pin for LED1 relay")
    ap.add_argument("--led2-pin", type=int, default=22, help="BCM pin for LED2 relay")
    ap.add_argument("--led3-pin", type=int, default=23, help="BCM pin for LED3 relay")
    ap.add_argument("--active-low", action="store_true", default=True, help="Relays are active-LOW (default)")
    ap.add_argument("--on-time", type=float, default=1.0, help="Seconds each LED stays ON during capture")
    ap.add_argument("--driver-settle", type=float, default=0.10, help="Seconds after enabling driver before LED on")
    ap.add_argument("--led-settle", type=float, default=0.30, help="Seconds after LED on before capture")
    ap.add_argument("--between-led-delay", type=float, default=0.30, help="Seconds after turning LED off before next")
    ap.add_argument("--out-dir", default="captures", help="Output folder for images")
    ap.add_argument("--prefix", default="sample", help="Filename prefix")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index counter")
    ap.add_argument("--repeat", type=int, default=1, help="Number of cycles to run")
    # Exposure/gain per LED (optional)
    '''ap.add_argument("--exp-led1", type=float, default=None, help="Exposure for LED1 (microseconds); omit to keep current")
    ap.add_argument("--exp-led2", type=float, default=None, help="Exposure for LED2 (microseconds)")
    ap.add_argument("--exp-led3", type=float, default=None, help="Exposure for LED3 (microseconds)")
    ap.add_argument("--gain-led1", type=float, default=None, help="Gain for LED1 (dB)")
    ap.add_argument("--gain-led2", type=float, default=None, help="Gain for LED2 (dB)")
    ap.add_argument("--gain-led3", type=float, default=None, help="Gain for LED3 (dB)")'''
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Relays
    driver = Relay(args.driver_pin, active_low=args.active_low, name="DRIVER")
    led1   = Relay(args.led1_pin,   active_low=args.active_low, name="LED1")
    led2   = Relay(args.led2_pin,   active_low=args.active_low, name="LED2")
    led3   = Relay(args.led3_pin,   active_low=args.active_low, name="LED3")
    '''leds = [(1, led1, args.exp_led1, args.gain_led1),
            (2, led2, args.exp_led2, args.gain_led2),
            (3, led3, args.exp_led3, args.gain_led3)]'''

    # Ensure all off initially
    for r in (driver, led1, led2, led3):
        r.off()

    # Camera
    system, cam, cam_list = init_pyspin_camera()
    print("[i] Camera initialized.")

    try:
        sample_idx = args.start_index
        for cycle in range(args.repeat):
            input("Press ENTER to start cycle (Ctrl+C to quit)... ") if args.repeat == 1 else print(f"[i] Starting cycle {cycle+1}/{args.repeat}")

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            for led_num, led_dev, exp_us, gain_db in leds:
                # 1) Enable driver/DIM relay
                driver.on()
                time.sleep(args.driver_settle)

                # 2) Turn on specific LED relay
                led_dev.on()
                time.sleep(args.led_settle)

                # 3) Lock manual exposure/gain (per LED if provided)
                set_manual_exposure_gain(cam, exposure_us=exp_us, gain_db=gain_db)

                # 4) Grab image
                arr = grab_pyspin_image(cam)
                fname = f"{args.prefix}_{sample_idx:03d}_LED{led_num}_{ts}.png"
                out_path = out_dir / fname
                cv2.imwrite(str(out_path), arr)
                print(f"[✓] Saved {out_path}")

                # 5) Turn off LED + driver
                led_dev.off()
                driver.off()
                time.sleep(args.between_led_delay)

            sample_idx += 1

        print("[i] All cycles complete.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")

    finally:
        # Ensure everything is off and GPIO released
        for r in (driver, led1, led2, led3):
            r.off()
            r.close()
        release_pyspin(system, cam, cam_list)
        print("[i] Cleaned up camera and GPIO.")

if __name__ == "__main__":
    main()
