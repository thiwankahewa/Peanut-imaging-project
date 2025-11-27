import cv2
import PySpin

# Initialize system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    print("No FLIR camera detected.")
    exit(1)

cam = cam_list.GetByIndex(0)
cam.Init()

cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureTime.SetValue(17800.0)  # microseconds
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
cam.Gain.SetValue(46.6)
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

cam.DeInit()
cam_list.Clear()
system.ReleaseInstance()
