# With this sample, you can obtain and save the depth map.
# 通过此示例，您可以获取并保存深度图。


import cv2

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d


class CaptureDepthMap(object):
    def __init__(self):
        self.camera = Camera()

    def capture_depth_map(self):
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))

        depth_map = frame3d.get_depth_map()
        depth_file = "dataset/DepthMap.tiff"
        cv2.imwrite(depth_file, depth_map.data())
        print("Capture and save the depth map: {}".format(depth_file))

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            self.capture_depth_map()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CaptureDepthMap()
    a.main()

# '''
#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\capture_depth_map.py
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
# Do you want the Stereo_Camera to capture 3D image ? Please input y/n to confirm:
# y
# Capture and save the depth map: DepthMap.tiff
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
# '''
