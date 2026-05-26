# With this sample, you can obtain and save 2D images, depth maps and point clouds
# simultaneously from multiple cameras.
# 通过此示例，您可以获取并保存二维图像、深度图和点云
# 同时从多个摄像机获取。

import os
import cv2
import threading
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


class CaptureThread(threading.Thread):
    def __init__(self, camera):
        threading.Thread.__init__(self)
        self.camera = camera

    def run(self):
        camera_info = CameraInfo()
        show_error(self.camera.get_camera_info(camera_info))
        print("Camera {} start capturing.".format(camera_info.ip_address))
        frame_all_2d_3d = Frame2DAnd3D()
        show_error(self.camera.capture_2d_and_3d(frame_all_2d_3d))

        # Save the obtained data with the set filenames.
        # color_file = "2DImage_" + camera_info.serial_number + ".png"
        # depth_file = "DepthMap_" + camera_info.serial_number + ".png"
        # point_cloud_file = "TexturedPointCloud_" + camera_info.serial_number + "ply"

        color_file = os.path.join(save_directory, f"2DImage_" + camera_info.serial_number + ".png")
        depth_file = os.path.join(save_directory, f"DepthMap_" + camera_info.serial_number + ".png")
        point_cloud_file = os.path.join(save_directory, f"TexturedPointCloud_" + camera_info.serial_number + "ply")

        color_image = frame_all_2d_3d.frame_2d().get_color_image()
        cv2.imwrite(color_file, color_image.data())
        print("Capture and save the 2D image:", color_file)

        depth_image = frame_all_2d_3d.frame_3d().get_depth_map()
        cv2.imwrite(depth_file, depth_image.data())
        print("Capture and save the depth map:", depth_file)

        show_error(
            frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY, point_cloud_file))
        print("Capture and save the textured point cloud:", point_cloud_file)

        self.camera.disconnect()
        print("Disconnected from the Stereo_Camera successfully.")


class MultipleCamerasCaptureSimultaneously(object):
    def __init__(self):
        self.cameras = find_and_connect_multi_camera()

    def connect_device_and_capture(self):
        if (len(self.cameras) == 0):
            print("No cameras connected.")
            return

        if not confirm_capture_3d():
            return

        threads = []
        for camera in self.cameras:
            capture_thread = CaptureThread(camera)
            threads.append(capture_thread)
            capture_thread.start()
        for thread in threads:
            thread.join()

    def main(self):
        self.connect_device_and_capture()


if __name__ == '__main__':
    a = MultipleCamerasCaptureSimultaneously()
    a.main()

#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\multiple_cameras_capture_simultaneously.py
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
# Enter the character 'c' to terminate adding devices
# 0
# Please enter the device index you want to connect:
# Enter the character 'c' to terminate adding devices
# c
# Do you want the Stereo_Camera to capture 3D image ? Please input y/n to confirm:
# y
# Camera 169.254.7.42 start capturing.
# Capture and save the 2D image: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\2DImage_NEM12238A4130015.png
# Capture and save the depth map: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\DepthMap_NEM12238A4130015.png
# Capture and save the textured point cloud: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\TexturedPointCloud_NEM12238A4130015ply
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
