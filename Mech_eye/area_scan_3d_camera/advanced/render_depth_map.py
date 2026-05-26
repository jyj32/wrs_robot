# With this sample, you can obtain and save the depth map.
# 通过此示例，您可以获取并保存深度图。

import os

import cv2
import numpy as np

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


# RenderedDepthMap 类: 该类的主要目的是与相机进行交互，捕获 3D 数据（深度图），并将其渲染为可视化图像。
class RenderedDepthMap(object):
    def __init__(self):
        self.camera = Camera()

    def render_depth_data(self, depth):
        if depth is None or depth.size == 0:
            return np.array([])

        mask = np.isfinite(depth).astype(np.uint8)
        min_depth_value, max_depth_value, _, _ = cv2.minMaxLoc(depth, mask)

        if np.isclose(max_depth_value - min_depth_value, 0):
            depth8U = depth.astype(np.uint8)
        else:
            depth8U = cv2.convertScaleAbs(depth, alpha=(255.0 / (min_depth_value - max_depth_value)),
                                          beta=((max_depth_value * 255.0) / (max_depth_value - min_depth_value) + 1))

        if depth8U.size == 0:
            return np.array([])

        colored_depth = cv2.applyColorMap(depth8U, cv2.COLORMAP_JET)
        colored_depth[depth8U == 0] = [0, 0, 0]

        return colored_depth

    def capture_rendered_depth_map(self):
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))

        rendered_depth_map = self.render_depth_data(frame3d.get_depth_map().data())

        rendered_depth_file = os.path.join(save_directory, f"RenderedDepthMap.tiff")
        # rendered_depth_file = "RenderedDepthMap.tiff"
        cv2.imwrite(rendered_depth_file, rendered_depth_map)
        print("Capture and save the rendered depth map: {}".format(rendered_depth_file))

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            self.capture_rendered_depth_map()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = RenderedDepthMap()
    a.main()

#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\render_depth_map.py
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
# Capture and save the rendered depth map: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\RenderedDepthMap.tiff
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
