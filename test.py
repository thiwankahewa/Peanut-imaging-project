import PySpin
import time

NUM_IMAGES = 4  # only capture 4 frames

def configure_exposure(cam):
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off automatic exposure mode
        cam.ExposureTime.SetValue(100)
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    
def configure_trigger(cam):
    try:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        
        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        cam.LineMode.SetValue(PySpin.LineMode_Output)
        cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)

def acquire_images(cam):
    nodemap = cam.GetNodeMap()

    # Start acquisition
    cam.BeginAcquisition()

    last_time = None

    for i in range(NUM_IMAGES):
        # Wait for next triggered frame
        image_result = cam.GetNextImage(1000)  # 1000 ms timeout

        if image_result.IsIncomplete():
            print("Image %d incomplete with image status %d ..." % (i, image_result.GetImageStatus()))
        else:
            now = time.time()
            if last_time is not None:
                dt = (now - last_time) * 1000  # ms
                print(f"Frame {i+1} captured. Δt = {dt:.2f} ms")
            else:
                print(f"Frame {i+1} captured.")

            last_time = now

            # Save image
            filename = f"image_{i+1}.jpg"
            image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            image_converted.Save(filename)
            print(f"Saved {filename}")

        # Release buffer
        image_result.Release()

    cam.EndAcquisition()

    # Disable trigger mode (restore to default)
    trigger_mode.SetIntValue(trigger_mode.GetEntryByName("Off").GetValue())
    print("Acquisition finished, trigger disabled.")


if __name__ == "__main__":
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No camera detected.")
    else:
        for i, cam in enumerate(cam_list):
            nodemap_tldevice = cam.GetTLDeviceNodeMap() #Retrieve TL device nodemap and print device information
            cam.Init()  #Camera becomes connected
            configure_exposure(cam)
            configure_trigger(cam)
            cam = None
            
        for i, cam in enumerate(cam_list):
            acquire_images(cam, nodemap_tldevice) # Acquire images
            cam.DeInit()  #ensure that devices clean up properly

    cam_list.Clear()
    system.ReleaseInstance()
