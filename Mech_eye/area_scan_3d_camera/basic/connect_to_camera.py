# With this sample program, you can connect to a Mech-Eye Industrial 3D Camera.

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *


class ConnectToCamera(object):
    def __init__(self):
        self.camera = Camera()

    def main(self):
        print("Discovering all available cameras...")
        camera_infos = Camera.discover_cameras()

        if len(camera_infos) == 0:
            print("No cameras found.")
            return

        # Display the information of all available cameras.
        for i in range(len(camera_infos)):
            print("Camera index :", i)
            print_camera_info(camera_infos[i])

        print("Please enter the index of the Stereo_Camera that you want to connect: ")
        input_index = 0

        # Enter the index of the Stereo_Camera to be connected and check if the index is valid.
        while True:
            input_index = input()
            if input_index.isdigit() and 0 <= int(input_index) < len(camera_infos):
                input_index = int(input_index)
                break
            print(
                "Input invalid! Please enter the index of the Stereo_Camera that you want to connect: ")

        error_status = self.camera.connect(camera_infos[input_index])
        if not error_status.is_ok():
            show_error(error_status)
            return

        print("Connected to the Stereo_Camera successfully.")

        self.camera.disconnect()
        print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = ConnectToCamera()
    a.main()

# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\connect_to_camera.py
# Discovering all available cameras...
# Camera index : 0
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
# Please enter the index of the Stereo_Camera that you want to connect:
# 0
# Connected to the Stereo_Camera successfully.
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
#
# '''
