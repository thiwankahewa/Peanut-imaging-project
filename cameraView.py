#!/usr/bin/env python3
import cv2
import PySpin

def main():
    # --- init camera ---
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() == 0:
        print("No FLIR camera found.")
        cams.Clear()
        system.ReleaseInstance()
        return

    cam = cams.GetByIndex(0)
    cam.Init()

    # Acquisition = Continuous
    nodemap = cam.GetNodeMap()
    acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    acq_cont = acq.GetEntryByName("Continuous")
    acq.SetIntValue(acq_cont.GetValue())
    processor = PySpin.ImageProcessor()
    # (optional) let camera auto-tune brightness for quick preview:
    # ExposureAuto = Continuous, GainAuto = Continuous
    try:
        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        exp_cont = exp_auto.GetEntryByName("Continuous")
        exp_auto.SetIntValue(exp_cont.GetValue())
        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        gain_cont = gain_auto.GetEntryByName("Continuous")
        gain_auto.SetIntValue(gain_cont.GetValue())
    except Exception:
        pass  # some models/nodes may differ; ignore for quick preview

    # --- start acquisition ---
    cam.BeginAcquisition()
    print("Streaming... press 'q' to quit")

    try:
        while True:
            img = cam.GetNextImage(1000)  # 1000 ms timeout
            if img.IsIncomplete():
                img.Release()
                continue

            # Convert to 8-bit for display
            frame = processor.Convert(img, PySpin.PixelFormat_Mono8).GetNDArray()
            img.Release()

            cv2.imshow("FLIR Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # --- cleanup ---
        cv2.destroyAllWindows()
        cam.EndAcquisition()
        cam.DeInit()
        cams.Clear()
        system.ReleaseInstance()
        print("Closed camera.")

if __name__ == "__main__":
    main()
