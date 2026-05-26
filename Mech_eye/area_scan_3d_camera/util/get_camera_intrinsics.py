# With this sample, you can obtain and print the Stereo_Camera intrinsic parameters.
# 通过此示例，您可以获取并打印相机固有参数。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_intrinsics


class GetCameraIntrinsics(object):
    def __init__(self):
        self.camera = Camera()
        self.intrinsics = CameraIntrinsics()

    def get_device_intrinsic(self):
        show_error(self.camera.get_camera_intrinsics(self.intrinsics))
        print_camera_intrinsics(self.intrinsics)




    def main(self):
        if find_and_connect(self.camera, 0):
            self.get_device_intrinsic()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = GetCameraIntrinsics()
    a.main()

#
# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\get_camera_intrinsics.py
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
# Texture Camera Matrix:
#     [2425.479641019078, 0, 959.7418484123396]
#     [0, 2425.328534633615, 632.7027593251646]
#     [0, 0, 1]
#
# Texture Camera Distortion Coefficients:
#     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0, k3: 0.0
#
# Depth Camera Matrix:
#     [2425.479641019078, 0, 959.7418484123396]
#     [0, 2425.328534633615, 632.7027593251646]
#     [0, 0, 1]
#
# Depth Camera Distortion Coefficients:
#     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0, k3: 0.0
#
# Rotation: From Depth Camera to Texture Camera:
#     [1.0, 0.0, 0.0]
#     [0.0, 1.0, 0.0]
#     [0.0, 0.0, 1.0]
#
# Translation From Depth Camera to Texture Camera:
#     X: 0.0mm, Y: 0.0mm, Z: 0.0mm
#
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''