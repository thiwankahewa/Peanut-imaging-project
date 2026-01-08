#!/usr/bin/env python3
import sys
import time
import cv2
import numpy as np
import PySpin


def configure_camera(cam: PySpin.CameraPtr):
    """
    Configure camera for continuous acquisition + auto exposure/gain (QuickSpin).
    Tries to be robust across FLIR models.
    """
    cam.Init()

    # --- Acquisition mode: Continuous ---
    if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    # --- Auto Exposure ---
    try:
        if cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
    except PySpin.SpinnakerException:
        pass

    # Optional: set exposure time limits (if supported) to avoid crazy values
    try:
        if hasattr(cam, "AutoExposureExposureTimeLowerLimit") and cam.AutoExposureExposureTimeLowerLimit.GetAccessMode() == PySpin.RW:
            cam.AutoExposureExposureTimeLowerLimit.SetValue(50.0)  # microseconds
        if hasattr(cam, "AutoExposureExposureTimeUpperLimit") and cam.AutoExposureExposureTimeUpperLimit.GetAccessMode() == PySpin.RW:
            cam.AutoExposureExposureTimeUpperLimit.SetValue(20000.0)  # microseconds
    except PySpin.SpinnakerException:
        pass

    # --- Auto Gain ---
    try:
        if cam.GainAuto.GetAccessMode() == PySpin.RW:
            cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
    except PySpin.SpinnakerException:
        pass

    # Optional: set gain limits (if supported)
    try:
        if hasattr(cam, "AutoGainLowerLimit") and cam.AutoGainLowerLimit.GetAccessMode() == PySpin.RW:
            cam.AutoGainLowerLimit.SetValue(0.0)
        if hasattr(cam, "AutoGainUpperLimit") and cam.AutoGainUpperLimit.GetAccessMode() == PySpin.RW:
            cam.AutoGainUpperLimit.SetValue(18.0)
    except PySpin.SpinnakerException:
        pass

    # --- Pixel format: prefer Mono8 for easiest OpenCV path ---
    try:
        if cam.PixelFormat.GetAccessMode() == PySpin.RW:
            # Mono8 is ideal; if not available, you can fall back to BayerRG8 etc.
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    except PySpin.SpinnakerException:
        pass

    # Buffer handling (helps reduce latency)
    try:
        s = cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            handling_mode.SetIntValue(newest_only.GetValue())
    except PySpin.SpinnakerException:
        pass


def get_frame_as_gray(cam: PySpin.CameraPtr, timeout_ms=1000) -> np.ndarray:
    """
    Acquire one image and return a grayscale numpy array.
    """
    img = cam.GetNextImage(timeout_ms)
    if img.IsIncomplete():
        img.Release()
        return None

    # If camera is Mono8 -> direct
    # Otherwise, convert to Mono8 using PySpin conversion
    if img.GetPixelFormat() != PySpin.PixelFormat_Mono8:
        img_conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        img.Release()
        img = img_conv

    w = img.GetWidth()
    h = img.GetHeight()
    data = img.GetData()  # bytes-like
    frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w)

    img.Release()
    return frame


def main():
    # ---------------- ArUco setup ----------------
    # Pick the dictionary you used to print markers:
    # DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_APRILTAG_36h11, etc.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # ---------------- FLIR camera setup ----------------
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("No FLIR/Spinnaker cameras found.")
        sys.exit(1)

    cam = cam_list.GetByIndex(0)

    try:
        configure_camera(cam)
        cam.BeginAcquisition()
        print("Streaming... Press 'q' to quit.")

        fps_t0 = time.time()
        fps_count = 0
        fps = 0.0

        while True:
            gray = get_frame_as_gray(cam)
            if gray is None:
                continue

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)

            # Draw overlay on BGR for colored annotations
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                # Put ID text near each marker (optional, drawDetectedMarkers already labels)
                for c, mid in zip(corners, ids.flatten()):
                    # c shape: (1, 4, 2)
                    pts = c[0]
                    x, y = int(pts[0][0]), int(pts[0][1])
                    cv2.putText(vis, f"ID:{mid}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # FPS counter (optional)
            fps_count += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_count / dt
                fps_count = 0
                fps_t0 = time.time()
            cv2.putText(vis, f"FPS:{fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            DISPLAY_WIDTH = 960   # change as needed (e.g., 800, 640)
            h, w = vis.shape[:2]
            scale = DISPLAY_WIDTH / w
            display = cv2.resize(vis, (int(w * scale), int(h * scale)))

            cv2.imshow("FLIR ArUco Live", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except PySpin.SpinnakerException as e:
        print("Spinnaker error:", e)

    finally:
        try:
            cam.EndAcquisition()
        except Exception:
            pass
        try:
            cam.DeInit()
        except Exception:
            pass
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
