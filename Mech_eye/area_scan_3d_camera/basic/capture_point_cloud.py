# With this sample, you can obtain and save the untextured and textured point clouds.
# 通过此示例，您可以获取并保存无纹理和有纹理的点云。
'''
未纹理化点云（Untextured Point Cloud）：仅包含空间坐标信息的点云，不含任何纹理（颜色或表面细节）。
纹理化点云（Textured Point Cloud）：包含空间坐标和与表面相关的纹理信息，通常是在 3D 扫描或模型重建过程中生成的。
'''

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d


class CapturePointCloud(object):
    def __init__(self):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()

    def capture_point_cloud(self):
        point_cloud_file = "dataset/PointCloud.ply"
        show_error(
            self.frame_all_2d_3d.frame_3d().save_untextured_point_cloud(FileFormat_PLY, point_cloud_file))
        print("Capture and save the untextured point cloud: {}.".format(
            point_cloud_file))

    def capture_textured_point_cloud(self):
        textured_point_cloud_file = "dataset/TexturedPointCloud.ply"
        show_error(self.frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY,
                                                                  textured_point_cloud_file))
        print("Capture and save the textured point cloud: {}".format(
            textured_point_cloud_file))

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            show_error(self.camera.capture_2d_and_3d(self.frame_all_2d_3d))
            self.capture_point_cloud()
            self.capture_textured_point_cloud()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CapturePointCloud()
    a.main()

# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\capture_point_cloud.py
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
# Capture and save the untextured point cloud: PointCloud.ply.
# Capture and save the textured point cloud: TexturedPointCloud.ply
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''
