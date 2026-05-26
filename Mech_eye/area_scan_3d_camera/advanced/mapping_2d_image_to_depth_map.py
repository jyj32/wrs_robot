# With this sample, you can generate untextured and textured point clouds from a masked 2D image and a depth map.
# 通过此示例，您可以从蒙版二维图像和深度图生成无纹理和有纹理的点云。
# 蒙版的作用是通过在深度图上遮挡特定区域来帮助生成更为精确的点云。
# 如果你只关心某些区域（例如你定义的 ROI 区域），那么掩膜将帮助你忽略其他部分。
# 相应的，白色区域代表着“无效”区域，在点云中会对应为空白（没有生成点）。

import os.path

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


class Mapping2DImageToDepthMap(object):
    def __init__(self):
        self.camera = Camera()

    def contains(self, row, col, roi):
        return roi[1] <= row <= roi[1] + roi[3] and roi[0] <= col <= roi[0] + roi[2]

    def generate_texture_mask(self, color: Color2DImage, roi1: ROI, roi2: ROI):
        mask = GrayScale2DImage()
        height = color.height()
        width = color.width()
        mask.resize(width, height)

        for i in range(height):
            for j in range(width):
                if not self.contains(i, j, roi1) and not self.contains(i, j, roi2):
                    mask.at(i, j).gray = 255
        return mask

    # 2D图像与深度图映射生成点云
    def mapping_2d_image_to_depth_map(self):
        # capture frame
        frame_all_2d_3d = Frame2DAnd3D()
        show_error(self.camera.capture_2d_and_3d(frame_all_2d_3d))
        color = frame_all_2d_3d.frame_2d().get_color_image()
        depth = frame_all_2d_3d.frame_3d().get_depth_map()
        intrinsics = CameraIntrinsics()
        show_error(self.camera.get_camera_intrinsics(intrinsics))

        roi1 = (color.width() / 5, color.height() / 5,
                color.width() / 2, color.height() / 2)
        roi2 = (color.width() * 2 / 5, color.height() * 2 /
                5, color.width() / 2, color.height() / 2)
        #
        #  Generate a mask of the following shape:
        #   ______________________________
        #  |                              |
        #  |                              |
        #  |   *****************          |
        #  |   *****************          |
        #  |   ************************   |
        #  |   ************************   |
        #  |          *****************   |
        #  |          *****************   |
        #  |                              |
        #  |______________________________|
        #
        mask = self.generate_texture_mask(color, roi1, roi2)

        points_xyz = UntexturedPointCloud()
        # get_point_cloud_after_mapping 方法使用深度图、掩膜和相机内参生成无纹理点云 points_xyz 。
        show_error(get_point_cloud_after_mapping(depth, mask, intrinsics, points_xyz))
        # point_cloud_file = "UntexturedPointCloud.ply"
        point_cloud_file = os.path.join(save_directory, f"UntexturedPointCloud.ply")
        show_error(Frame3D.save_point_cloud(points_xyz, FileFormat_PLY, point_cloud_file))
        print("Save the untextured point cloud to file:", point_cloud_file)

        # generate colored point cloud
        points_xyz_bgr = TexturedPointCloud()
        show_error(get_point_cloud_after_mapping(depth, mask, color, intrinsics, points_xyz_bgr))
        point_cloud_file = os.path.join(save_directory, f"TexturedPointCloud.ply")
        show_error(Frame2DAnd3D.save_point_cloud(points_xyz_bgr, FileFormat_PLY, point_cloud_file))
        print("Save the textured point cloud to file:", point_cloud_file)

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            self.mapping_2d_image_to_depth_map()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = Mapping2DImageToDepthMap()
    a.main()

#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\mapping_2d_image_to_depth_map.py
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
# Save the untextured point cloud to file: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\UntexturedPointCloud.ply
# Save the textured point cloud to file: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\TexturedPointCloud.ply
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
