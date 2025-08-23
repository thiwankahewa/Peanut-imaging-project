import PySpin
import sys
import time

def configure_exposure(cam):
    print('*** CONFIGURING EXPOSURE ***\n')
    
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off automatic exposure mode
        cam.ExposureTime.SetValue(100)
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    
def configure_trigger(cam):
    print('*** CONFIGURING TRIGGER ***\n')
    
    try:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)

def acquire_images(cam, nodemap_tldevice):
    print('*** IMAGE ACQUISITION ***\n')
    
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)# Set acquisition mode to continuous
    cam.BeginAcquisition()
    
    device_serial_number = ''
    node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
    if PySpin.IsReadable(node_device_serial_number):
        device_serial_number = node_device_serial_number.GetValue()  # Retrieve device serial number for logging
    
    processor = PySpin.ImageProcessor()  # Create ImageProcessor instance for post processing images
    #processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)   # Set default image processor color processing method(for color cameras)
    
    for i in range(4):
        while True:
            try:
                
                start_time = time.time()
                image_result = cam.GetNextImage(1000)  #  Retrieve next received image
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())   #  Ensure image completion
                    continue

                else:
                    filename = f'Acquisition-{device_serial_number}-{i + 1}.jpg'
                    image_result.Save(filename)
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

def main():
    result = True
    system = PySpin.System.GetInstance()     # Everything originates with the system object
    cam_list = system.GetCameras()       # Retrieve list of cameras from the system
    num_cams = cam_list.GetSize()   # Get number of cameras# Get number of cameras
    
    if num_cams == 0:
        cam_list.Clear()    # Clear camera list before releasing system
        system.ReleaseInstance()     # Release system instance
        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    
    
    for i, cam in enumerate(cam_list):
        nodemap_tldevice = cam.GetTLDeviceNodeMap() #Retrieve TL device nodemap and print device information
        cam.Init()  #Camera becomes connected
        configure_exposure(cam)
        configure_trigger(cam)
        
    for i, cam in enumerate(cam_list):
        acquire_images(cam, nodemap_tldevice) # Acquire images
        
        cam.DeInit()  #ensure that devices clean up properly
        
    
    cam_list.Clear()
    system.ReleaseInstance()
    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

        
    