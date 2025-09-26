import PySpin
import sys
import time

def configure_camera(cam):
    try:
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)# Set acquisition mode to continuous
        #cam.AcquisitionFrameRateEnable.SetValue(True)  # Turn on frame rate enable
        #cam.AcquisitionFrameRate.SetValue(4)  # Set frame rate to 30
        
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off automatic exposure mode
        cam.ExposureTime.SetValue(100)
        
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
    cam.BeginAcquisition()
    
    images = [] 
    processor = PySpin.ImageProcessor()  # Create ImageProcessor instance for post processing images
    #processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)   # Set default image processor color processing method(for color cameras)
    print("Waiting for trigger...")
    
    for i in range(10):
        while True:
            try:
                
                start_time = time.time()
                image_result = cam.GetNextImage(1000)  #  Retrieve next received image
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())   #  Ensure image completion
                    continue

                else:
                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
                    images.append(image_converted)
                    image_result.Release()   #  Release image
                    end_time = time.time()
                    time_diff = end_time - start_time
                    print(f'Time difference for image acquisition: {time_diff:.6f} seconds, Image count: {i + 1}')
                    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                    break
                    
            except PySpin.SpinnakerException as ex:
                if "[-1011]" in str(ex):  # timeout / buffer error
                    print(f"Timeout waiting for image......")
                    continue  
                else:
                    print(f"Unexpected error: {ex}")
                    break  
                
    cam.EndAcquisition()
    
    for i, image in enumerate(images):
        filename = f'cam1-{i+1}.jpg'
        image.Save(filename)

def main():
    result = True
    system = PySpin.System.GetInstance()     # Everything originates with the system object
    cam_list = system.GetCameras()       # Retrieve list of cameras from the system
    num_cams = cam_list.GetSize()   # Get number of cameras# Get number of cameras
    
    if num_cams == 0:
        cam_list.Clear()    # Clear camera list before releasing system
        system.ReleaseInstance()     # Release system instance
        print('Not enough cameras!')
        return False
    
    
    for i,cam in enumerate(cam_list):
        cam.Init()  #Camera becomes connected
        configure_camera(cam)
        
    for i,cam in enumerate(cam_list):
        acquire_images(cam) # Acquire images
        cam.DeInit()  #ensure that devices clean up properly
        
    
    cam_list.Clear()
    system.ReleaseInstance()
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

        
    