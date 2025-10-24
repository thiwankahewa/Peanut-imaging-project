import sys
import time
import argparse
import pathlib
import datetime
import serial
import serial.tools.list_ports

# Try PySpin first; fallback to OpenCV if not present
USE_PYSPIN = False
cam = None
try:
    import PySpin  # FLIR Spinnaker
    USE_PYSPIN = True
except Exception:
    USE_PYSPIN = False
    import cv2

def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def init_serial(port, baud=115200, timeout=5.0):
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    # Give Arduino time to reset (some boards reset on open)
    time.sleep(2.0)
    # Drain any boot text
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

# ---------- Camera helpers ----------
def init_pyspin():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR camera found via PySpin.")
    cam = cam_list.GetByIndex(0)
    cam.Init()
    # Single frame mode
    s_node_map = cam.GetNodeMap()
    acq_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("AcquisitionMode"))
    if not PySpin.IsAvailable(acq_mode) or not PySpin.IsWritable(acq_mode):
        raise RuntimeError("AcquisitionMode not writable")
    acq_mode_single = acq_mode.GetEntryByName("SingleFrame")
    acq_mode.SetIntValue(acq_mode_single.GetValue())
    return system, cam, cam_list

def grab_pyspin_image(cam):
    cam.BeginAcquisition()
    img = cam.GetNextImage(1000)  # 1000 ms timeout
    if img.IsIncomplete():
        cam.EndAcquisition()
        raise RuntimeError(f"Incomplete image: {img.GetImageStatus()}")
    # Convert to 8-bit
    converted = img.Convert(PySpin.PixelFormat_Mono8, PySpin.DIRECTIONAL_FILTER)
    data = converted.GetNDArray()
    img.Release()
    cam.EndAcquisition()
    return data

def save_numpy_png(numpy_img, path):
    # Use OpenCV to write PNG even if we captured with PySpin
    import cv2
    cv2.imwrite(str(path), numpy_img)

def release_pyspin(system, cam, cam_list):
    try:
        cam.DeInit()
    except Exception:
        pass
    cam_list.Clear()
    system.ReleaseInstance()

def init_opencv(index=0, width=None, height=None):
    cap = cv2.VideoCapture(index)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Failed to open OpenCV camera.")
    return cap

def grab_opencv_image(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("OpenCV capture failed.")
    return frame

def save_bgr_png(bgr_img, path):
    cv2.imwrite(str(path), bgr_img)

# ---------- Main sequence ----------
def wait_for_ready(ser, expected_idx, timeout=5.0):
    """Wait for 'READY n' line from Arduino."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        # print("SER>", line)
        if line == f"READY {expected_idx}":
            return True
        if line.startswith("ERR"):
            raise RuntimeError(f"Arduino error: {line}")
    raise TimeoutError(f"Timeout waiting READY {expected_idx}")

def send_cmd(ser, cmd):
    ser.write((cmd + "\n").encode())

def run_sequence(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serial
    ser = init_serial(args.port, baud=args.baud, timeout=5.0)
    print(f"[i] Connected to {args.port}")

    # Camera
    if USE_PYSPIN:
        print("[i] Using PySpin (FLIR)")
        system, cam, cam_list = init_pyspin()
    else:
        print("[i] Using OpenCV fallback")
        cap = init_opencv(index=args.camera_index, width=args.width, height=args.height)

    sample_idx = args.start_index
    try:
        while True:
            input("Press ENTER to start a 3-LED capture cycle (Ctrl+C to quit)... ")

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for led in (1, 2, 3):
                # Turn only this LED on
                send_cmd(ser, f"SET {led}")
                wait_for_ready(ser, led, timeout=10.0)

                # Capture
                fname = f"sample_{sample_idx:03d}_LED{led}_{ts}.png"
                out_path = out_dir / fname

                if USE_PYSPIN:
                    img = grab_pyspin_image(cam)   # numpy uint8 (mono)
                    save_numpy_png(img, out_path)
                else:
                    frame = grab_opencv_image(cap) # BGR
                    save_bgr_png(frame, out_path)

                print(f"[✓] Saved {out_path}")

            # All off (optional; Arduino already keeps only-one-on. This ensures off after cycle)
            send_cmd(ser, "ALL OFF")
            print("[i] Cycle complete.")

            # Repeat?
            ans = input("Repeat this 3-LED cycle? [y/N]: ").strip().lower()
            if ans != "y":
                break
            sample_idx += 1

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        # Cleanup
        try:
            send_cmd(ser, "ALL OFF")
        except Exception:
            pass
        ser.close()
        if USE_PYSPIN:
            release_pyspin(system, cam, cam_list)
        else:
            cap.release()
        print("[i] Closed resources.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-LED capture sequence controller")
    parser.add_argument("--port", required=False, default=None, help="Serial port to Arduino (e.g., /dev/ttyACM0, COM3)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out-dir", default="captures")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV device index if PySpin not available")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    if args.port is None:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found. Plug in Arduino and try again.")
            sys.exit(1)
        print("Detected ports:", ports)
        # Pick the first by default; adjust if needed
        args.port = ports[0]
        print(f"Using: {args.port}")

    run_sequence(args)
