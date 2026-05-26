# With this sample, you can set multiple exposure times, and then obtain and save the point cloud.
# 通过此示例，您可以设置多次曝光时间，然后获取并保存点云。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d


class CapturePointCloudHDR(object):
    def __init__(self):
        self.camera = Camera()
        self.frame_2d_and_3d = Frame2DAnd3D()

    def capture_point_cloud(self):
        point_cloud_file = "dataset/PointCloud.ply"
        show_error(self.frame_2d_and_3d.frame_3d().save_untextured_point_cloud(FileFormat_PLY, point_cloud_file))
        print("Capture and save the untextured point cloud to {}.".format(point_cloud_file))

    def capture_textured_point_cloud(self):
        textured_point_cloud_file = "dataset/TexturedPointCloud.ply"
        show_error(self.frame_2d_and_3d.save_textured_point_cloud(FileFormat_PLY, textured_point_cloud_file))
        print("Capture and save the textured point cloud to {}".format(textured_point_cloud_file))

    def capture_point_cloud_hdr(self):
        # Set 3D Exposure Sequence.
        current_user_set = self.camera.current_user_set()
        error = current_user_set.set_float_array_value(Scanning3DExposureSequence.name, [5, 10])
        show_error(error)
        self.capture_point_cloud()
        self.capture_textured_point_cloud()

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            show_error(self.camera.capture_2d_and_3d(self.frame_2d_and_3d))
            self.capture_point_cloud_hdr()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CapturePointCloudHDR()
    a.main()


# '''
#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\capture_point_cloud_hdr.py
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
# Capture and save the untextured point cloud to PointCloud.ply.
# Capture and save the textured point cloud to TexturedPointCloud.ply
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''
