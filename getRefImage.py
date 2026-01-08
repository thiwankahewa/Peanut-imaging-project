import sys
import cv2
import PySpin

# Initialize system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

num_cams = cam_list.GetSize()
if num_cams == 0:
    print("No FLIR camera detected.")

    # --- CLEANUP before exit ---
    cam_list.Clear()
    del cam_list
    system.ReleaseInstance()
    sys.exit(1)

# Get first camera
cam = cam_list.GetByIndex(0)

try:
    cam.Init()

    # Camera settings
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
    #cam.ExposureTime.SetValue(17800.0)  # microseconds
    cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
    #cam.Gain.SetValue(46.6)
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

    # Capture one image
    cam.BeginAcquisition()
    processor = PySpin.ImageProcessor()

    img = cam.GetNextImage(1000)
    if img.IsIncomplete():
        print("Incomplete image.")
    else:
        arr = processor.Convert(img, PySpin.PixelFormat_Mono16).GetNDArray()
        cv2.imwrite("reference.png", arr)
        print("Saved reference.png")

    img.Release()
    cam.EndAcquisition()

finally:
    # De-init and clean everything in the right order
    try:
        if cam.IsInitialized():
            cam.DeInit()
    except Exception:
        # In case IsInitialized() doesn't exist or something odd happens
        pass

    # Drop references BEFORE releasing system
    del cam
    cam_list.Clear()
    del cam_list

    system.ReleaseInstance()
