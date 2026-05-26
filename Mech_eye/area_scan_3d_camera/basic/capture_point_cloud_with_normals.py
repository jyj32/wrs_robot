# With this sample, you can calculate normals and save the point cloud with normals. The normals can be calculated on the Stereo_Camera or the computer.
# 通过此示例，您可以计算法线并保存包含法线的点云。可以在相机或计算机上计算法线。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d


class CapturePointCloudWithNormals(object):
    def __init__(self):
        self.camera = Camera()
        self.frame_3d = Frame3D()

    # Calculate the normals of the points on the Stereo_Camera and save the point cloud with normals to file
    # 计算相机上点的法线，并将包含法线的点云保存到文件

    def capture_point_cloud_with_normals_calculated_on_camera(self):
        point_cloud_file = "dataset/PointCloud_111.ply"
        if self.camera.capture_3d_with_normal(self.frame_3d).is_ok():
            show_error(
                self.frame_3d.save_untextured_point_cloud_with_normals(FileFormat_PLY, point_cloud_file))
            return True
        else:
            print("Failed to capture the point cloud.")
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")
            return False

    # Calculate the normals of the points on the computer and save the point cloud with normals to file
    # 计算电脑上点的法线，并将包含法线的点云保存到文件

    def capture_point_cloud_with_normals_calculated_locally(self):
        point_cloud_file = "dataset/PointCloud_222.ply"
        if self.camera.capture_3d(self.frame_3d).is_ok():
            show_error(
                self.frame_3d.save_untextured_point_cloud_with_normals(FileFormat_PLY, point_cloud_file))
            return True
        else:
            print("Failed to capture the point cloud.")
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")
            return False

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            if not self.capture_point_cloud_with_normals_calculated_on_camera():
                return
            if not self.capture_point_cloud_with_normals_calculated_locally():
                return
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CapturePointCloudWithNormals()
    a.main()

# '''
#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\capture_point_cloud_with_normals.py
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
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''
