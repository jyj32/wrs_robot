# With this sample, you can obtain and save the 2D image.
# 通过此示例，您可以获取并保存2D图像。

import cv2

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class Capture2DImage(object):
    def __init__(self):
        self.camera = Camera()

    def capture_2d_image(self):
        frame_2d = Frame2D()
        show_error(self.camera.capture_2d(frame_2d))
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()

        file_name = "dataset/2DImage.png"
        cv2.imwrite(file_name, image2d.data())
        print("Capture and save the 2D image: {}".format(file_name))

    def main(self):
        if find_and_connect(self.camera):
            self.capture_2d_image()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = Capture2DImage()
    a.main()

# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\capture_2d_image.py
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
# Capture and save the 2D image: 2DImage.png
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''
