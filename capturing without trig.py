import PySpin
import sys
import time
import numpy as np

exposure_time = [3000, 100]

def configure_exposure(cam, exposure_time):
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off automatic exposure mode
        cam.ExposureTime.SetValue(exposure_time) 
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    
def configure_trigger(cam):                   #turn off the trigger
    try:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
      
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)

def acquire_images(cam, nodemap_tldevice):
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)# Set acquisition mode to continuous
    cam.BeginAcquisition()
    
    device_serial_number = ''
    node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
    if PySpin.IsReadable(node_device_serial_number):
        device_serial_number = node_device_serial_number.GetValue()  # Retrieve device serial number for logging
    
    processor = PySpin.ImageProcessor()  # Create ImageProcessor instance for post processing images
    #processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)   # Set default image processor color processing method(for color cameras)
    
    images = [] 
    
    for i in range(4):
        try:
            
            start_time = time.time()
            image_result = cam.GetNextImage(1000)  #  Retrieve next received image
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())   #  Ensure image completion
                continue

            else:
                image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
                images.append(image_converted)
                end_time = time.time()
                time_diff = end_time - start_time
                print(f'Time difference for image acquisition: {time_diff:.6f} seconds, Image count: {i + 1}')
                image_result.Release()
                
        except PySpin.SpinnakerException as ex:
            if "[-1011]" in str(ex):  # timeout / buffer error
                print(f"Timeout waiting for image......")
                continue  
            else:
                print(f"Unexpected error: {ex}")
                break  
            
    cam.EndAcquisition()
    
    for i, image in enumerate(images):
        filename = f'Acquisition-{device_serial_number}-{i+1}.jpg'
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
        input('Done! Press Enter to exit...')
        return False
    
    nodemap_list = []
    for i, cam in enumerate(cam_list):
        nodemap_tldevice = cam.GetTLDeviceNodeMap() #Retrieve TL device nodemap and print device information
        nodemap_list.append(nodemap_tldevice)
        cam.Init()  #Camera becomes connected
        configure_exposure(cam, exposure_time[i])
        configure_trigger(cam)
        
    for i, cam in enumerate(cam_list):
        acquire_images(cam, nodemap_list[i]) # Acquire images
        
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

        
    