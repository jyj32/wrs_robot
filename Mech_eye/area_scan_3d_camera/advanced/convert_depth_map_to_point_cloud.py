# With this sample, you can generate a point cloud from the depth map and save the point cloud.
# 通过此示例，您可以从深度图生成点云并保存点云。
# 这段代码演示了如何从 3D 相机的深度图数据生成点云，
# 并将其保存为 .ply 文件，适用于 3D 数据处理和可视化应用。

import os
import numpy as np
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


class ConvertDepthMapToPointCloud(object):
    def __init__(self):
        self.camera = Camera()

    def convert_depth_map_to_point_cloud(self, depth: DepthMap, intrinsics: CameraIntrinsics,
                                         xyz: UntexturedPointCloud):
        xyz.resize(depth.width(), depth.height())

        for i in range(depth.width() * depth.height()):
            row = int(i / depth.width())
            col = int(i - row * depth.width())
            xyz[i].z = depth[i].z
            xyz[i].x = float((col - intrinsics.depth.camera_matrix.cx)
                             * depth[i].z / intrinsics.depth.camera_matrix.fx)
            xyz[i].y = float((row - intrinsics.depth.camera_matrix.cy)
                             * depth[i].z / intrinsics.depth.camera_matrix.fy)

    def capture_cloud_from_depth(self):
        camera_info = CameraInfo()
        show_error(self.camera.get_camera_info(camera_info))
        print_camera_info(camera_info)

        if not confirm_capture_3d():
            return

        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))
        depth = frame3d.get_depth_map()
        intrinsics = CameraIntrinsics()
        show_error(self.camera.get_camera_intrinsics(intrinsics))

        point_cloud = UntexturedPointCloud()
        self.convert_depth_map_to_point_cloud(depth, intrinsics, point_cloud)
        point_cloud_file = os.path.join(save_directory, f"UntexturedPointCloud.ply")
        # point_cloud_file = "dataset/UntexturedPointCloud.ply"
        Frame3D.save_point_cloud(point_cloud, FileFormat_PLY, point_cloud_file)

        print("The point cloud contains:", point_cloud.width()
              * point_cloud.height(), "data points.")
        print("Save the point cloud to file:", point_cloud_file)

    def main(self):
        if find_and_connect(self.camera):
            self.capture_cloud_from_depth()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = ConvertDepthMapToPointCloud()
    a.main()

#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\convert_depth_map_to_point_cloud.py
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
# Do you want the Stereo_Camera to capture 3D image ? Please input y/n to confirm:
# y
# The point cloud contains: 2304000 data points.
# Save the point cloud to file: G:/mecheye_python_samples/area_scan_3d_camera/advanced/dataset\UntexturedPointCloud.ply
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
