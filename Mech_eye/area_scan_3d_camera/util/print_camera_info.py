# With this sample, you can obtain and print the Stereo_Camera information, such as model, serial number, firmware version, and temperatures.
# 通过此示例，您可以获取并打印相机信息，例如型号、序列号、固件版本和温度。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_info, print_camera_status


class PrintCameraInfo(object):
    def __init__(self):
        self.camera = Camera()
        self.camera_info = CameraInfo()
        self.camera_status = CameraStatus()

    def print_device_info(self):
        show_error(self.camera.get_camera_info(self.camera_info))
        print_camera_info(self.camera_info)
        show_error(self.camera.get_camera_status(self.camera_status))
        print_camera_status(self.camera_status)

    def main(self):
        if find_and_connect(self.camera):
            self.print_device_info()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = PrintCameraInfo()
    a.main()


#
# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\print_camera_info.py
# Find Mech-Eye Industrial 3D Cameras...
# Mech-Eye device index : 0
# .............................
# Camera Model Name:           Mech-Eye PRO M
# Camera Serial Number:        NEM12238A4130015
# Camera IP Address:           169.254.7.42
# Camera Subnet Mask:          255.255.0.0
# Camera IP Assignment Method: LLA
# Hardware Version:            V4.1.0
# Firmware Version:            V2.4.0
# .............................
#
# Please enter the device index you want to connect:
# 0
# Connect Mech-Eye Industrial 3D Camera Successfully.
# .............................
# Camera Model Name:           Mech-Eye PRO M
# Camera Serial Number:        NEM12238A4130015
# Camera IP Address:           169.254.7.42
# Camera Subnet Mask:          255.255.0.0
# Camera IP Assignment Method: LLA
# Hardware Version:            V4.1.0
# Firmware Version:            V2.4.0
# .............................
#
# .....Camera Temperature.....
# CPU :               33.0°C
# Projector Module:   28.0°C
# ............................
#
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
#
# '''