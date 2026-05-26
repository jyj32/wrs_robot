# With this sample, you can set the "Point Cloud Processing" parameters.
# 通过此示例，您可以设置“点云处理”参数。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_info


class SetPointCloudProcessingParameters(object):
    def __init__(self):
        self.camera = Camera()

    def set_point_cloud_processing_parameters(self):
        # Obtain the basic information of the connected Stereo_Camera.
        cameraInfo = CameraInfo()
        show_error(self.camera.get_camera_info(cameraInfo))
        print_camera_info(cameraInfo)

        current_user_set = self.camera.current_user_set()

        # Set the "Point Cloud Processing" parameters, and then obtain the parameter values to check if the setting was successful.
        # 使用 set_enum_value 方法设置点云处理的相关参数：
        # PointCloudSurfaceSmoothing：表面平滑
        # PointCloudNoiseRemoval：噪声去除
        # PointCloudOutlierRemoval：离群点去除
        # PointCloudEdgePreservation：边缘保留

        error = current_user_set.set_enum_value(PointCloudSurfaceSmoothing.name, PointCloudSurfaceSmoothing.Value_Normal)
        show_error(error)
        error = current_user_set.set_enum_value(PointCloudNoiseRemoval.name, PointCloudNoiseRemoval.Value_Normal)
        show_error(error)
        error = current_user_set.set_enum_value(PointCloudOutlierRemoval.name, PointCloudOutlierRemoval.Value_Normal)
        show_error(error)
        error = current_user_set.set_enum_value(PointCloudEdgePreservation.name, PointCloudEdgePreservation.Value_Normal)
        show_error(error)

        error, surface_smoothing = current_user_set.get_enum_value_string(PointCloudSurfaceSmoothing.name)
        show_error(error)
        error, noise_removal = current_user_set.get_enum_value_string(PointCloudNoiseRemoval.name)
        show_error(error)
        error, outlier_removal = current_user_set.get_enum_value_string(PointCloudOutlierRemoval.name)
        show_error(error)
        error, edge_preservation = current_user_set.get_enum_value_string(PointCloudEdgePreservation.name)
        show_error(error)


        # Point Cloud Surface Smoothing（点云表面平滑）：
        # 0: Off：关闭平滑处理。
        # 1: Weak：弱平滑处理。
        # 2: Normal：正常平滑处理。
        # 3: Strong：强平滑处理。
        # 这个参数影响点云数据的表面平滑程度，数值越高，表面越平滑。
        #
        # Point Cloud Noise Removal（点云噪声去除）：
        # 0: Off：关闭噪声去除处理。
        # 1: Weak：弱噪声去除。
        # 2: Normal：正常噪声去除。
        # 3: Strong：强噪声去除。
        # 该参数决定了噪声去除的强度，数值越高，去除的噪声越多。
        #
        # Point Cloud Outlier Removal（点云离群点去除）：
        # 0: Off：关闭离群点去除处理。
        # 1: Weak：弱离群点去除。
        # 2: Normal：正常离群点去除。
        # 3: Strong：强离群点去除。
        # 这个参数影响离群点的去除程度，数值越高，去除的离群点越多。
        #
        # Point Cloud Edge Preservation（点云边缘保留）：
        # 0: Sharp：保留尖锐边缘。
        # 1: Normal：正常边缘保留。
        # 2: Smooth：平滑边缘保留。
        # 这个参数决定了在进行平滑处理时，是否保留边缘的锐度。数值越高，边缘越平滑。
        print("Point Cloud Surface Smoothing:", surface_smoothing, "(0: Off, 1: Weak, 2: Normal, 3: Strong)")
        print("Point Cloud Noise Removal:", noise_removal, "(0: Off, 1: Weak, 2: Normal, 3: Strong)")
        print("Point Cloud Outlier Removal:", outlier_removal, "(0: Off, 1: Weak, 2: Normal, 3: Strong)")
        print("Point Cloud Edge Preservation:", edge_preservation, "(0: Sharp, 1: Normal, 2: Smooth)")

        # Save all the parameter settings to the currently selected user set.
        # 将所有参数设置保存到当前选定的用户集。
        show_error(current_user_set.save_all_parameters_to_device())
        print("\nSave the current parameter settings to the selected user set..")

    def main(self):
        if find_and_connect(self.camera):
            self.set_point_cloud_processing_parameters()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = SetPointCloudProcessingParameters()
    a.main()

#
# '''
# # D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\set_point_cloud_processing_parameters.py
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
# Point Cloud Surface Smoothing: Normal (0: Off, 1: Weak, 2: Normal, 3: Strong)
# Point Cloud Noise Removal: Normal (0: Off, 1: Weak, 2: Normal, 3: Strong)
# Point Cloud Outlier Removal: Normal (0: Off, 1: Weak, 2: Normal, 3: Strong)
# Point Cloud Edge Preservation: Normal (0: Sharp, 1: Normal, 2: Smooth)
#
# Save the current parameter settings to the selected user set..
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
#
# '''
