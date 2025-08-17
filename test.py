import PySpin
import sys
import os

NUM_IMAGES = 10  # number of images to grab

def configure_exposure(cam):
    print('*** CONFIGURING EXPOSURE ***\n')
    
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # Turn off automatic exposure mode
        cam.ExposureTime.SetValue(500000)
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    
def configure_trigger(cam):
    print('*** CONFIGURING TRIGGER ***\n')
    
    try:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        
        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        cam.V3_3Enable.SetValue(True)  
        cam.LineMode.SetValue(PySpin.LineMode_Output)
        cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False


def acquire_images(cam, nodemap, nodemap_tldevice):
    print('*** IMAGE ACQUISITION ***\n')
    
    try:
        result = True
        
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)   # Set acquisition mode to continuous
        
        cam.BeginAcquisition()
        
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()  # Retrieve device serial number for logging
        
        processor = PySpin.ImageProcessor()  # Create ImageProcessor instance for post processing images
        #processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)   # Set default image processor color processing method(for color cameras)
        
        for i in range(NUM_IMAGES):
            try:
                image_result = cam.GetNextImage(1000)  #  Retrieve next received image

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())   #  Ensure image completion

                else:
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))
                    pixel_format = image_result.GetPixelFormat()
                    print("Pixel Format:", pixel_format)     #if the camera is 8 bit pass the following conversion          
                    
                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)   #  Convert image to mono 8

                    filename = 'Acquisition-%s-%d.jpg' % (device_serial_number, i)
                    image_converted.Save(filename)
                    print('Image saved at %s' % filename)
                    image_result.Release()   #  Release image

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        cam.EndAcquisition()
        
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def print_device_info(nodemap):
    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def main():
    try:    #Ensure that have permission to write to current folder.
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False
    
    test_file.close()
    os.remove(test_file.name)
    
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
        result &= print_device_info(nodemap_tldevice)   #prints the device information of the camera from the transport layer
        cam.Init()  #Camera becomes connected
        nodemap = cam.GetNodeMap()  # Retrieve GenICam nodemap
        configure_exposure(cam)
        configure_trigger(cam)
        
        is_stereo_camera = PySpin.ImageUtilityStereo.IsStereoCamera(cam)
        if is_stereo_camera:
            print('Camera is a stereo camera')
        else:
            print('Camerais not a valid BX camera. Skipping...' )
        
    for i, cam in enumerate(cam_list):
        result &= acquire_images(cam, nodemap, nodemap_tldevice) # Acquire images
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

        
    