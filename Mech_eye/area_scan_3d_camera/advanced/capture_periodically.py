# With this sample, you can obtain and save 2D images, depth maps and point clouds
# periodically for the specified duration from a Stereo_Camera.
# 在指定的时间内定期从相机获取。

import os
import time
import cv2

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

# Set the Stereo_Camera capture interval to 10 seconds and the total duration of image capturing to 5 minutes.
capture_time = 5  # minutes
capture_period = 10  # seconds

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


# 定期捕获类 CapturePeriodically
class CapturePeriodically(object):
    def __init__(self):
        self.camera = Camera()

    def capture_timed_and_periodically(self):
        frame_all_2d_3d = Frame2DAnd3D()
        show_error(self.camera.capture_2d_and_3d(frame_all_2d_3d))
        start = time.time()
        capture_count = 0

        # Perform image capturing periodically according to the set interval for the set total duration.
        # 根据设置的间隔定期执行图像捕获，持续的总时长为设置的总时长。
        while time.time() - start < capture_time * 60:
            before = time.time()
            print("Start capturing.")

            show_error(self.camera.capture_2d_and_3d(frame_all_2d_3d))
            capture_count = capture_count + 1

            # Save the obtained data with the set filenames.
            # 使用设置的文件名保存获取的数据。

            # 从捕获的二维图像中提取彩色图像，并保存为.png 文件。
            color_map = frame_all_2d_3d.frame_2d().get_color_image()
            # color_file = "2DImage_" + str(capture_count) + ".png"
            color_file = os.path.join(save_directory, f"2DImage_{capture_count}.png")
            cv2.imwrite(color_file, color_map.data())

            # 提取深度图数据并保存为 .png 文件。
            depth_map = frame_all_2d_3d.frame_3d().get_depth_map()
            # depth_file = "DepthMap_" + str(capture_count) + ".png"
            depth_file = os.path.join(save_directory, f"DepthMap_{capture_count}.png")
            cv2.imwrite(depth_file, depth_map.data())

            # 保存点云数据为.ply 文件，包含纹理信息。
            # point_cloud_file = "TexturedPointCloud_" + str(capture_count) + ".ply"
            point_cloud_file = os.path.join(save_directory, f"TexturedPointCloud_{capture_count}.ply")
            show_error(frame_all_2d_3d.save_textured_point_cloud(FileFormat_PLY, point_cloud_file))
            print("Paused capturing.")

            after = time.time()
            time_used = after - before
            if time_used < capture_period:
                time.sleep(capture_period - time_used)
            else:
                print(
                    "The actual capture time is longer than the set capture interval. Please increase the capture interval.")
                # 实际抓拍时间比设置的抓拍间隔长，请增大抓拍间隔。

            time_remaining = int(capture_time * 60 - (time.time() - start))
            print("Remaining time: {0} minutes {1} seconds".format(
                int(time_remaining / 60), time_remaining % 60))

        print("Capturing for {} minutes is completed.".format(capture_time))

    def main(self):
        if find_and_connect(self.camera):
            if not confirm_capture_3d():
                return
            self.capture_timed_and_periodically()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CapturePeriodically()
    a.main()

#
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\capture_periodically.py
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
# Start capturing.
# Paused capturing.
# Remaining time: 4 minutes 49 seconds
# Start capturing.
# Paused capturing.
# Remaining time: 4 minutes 39 seconds
# Start capturing.
# Paused capturing.
# Remaining time: 4 minutes 29 seconds
# Start capturing.
# Paused capturing.
# Remaining time: 4 minutes 19 seconds
# Start capturing.
# Paused capturing.
# Remaining time: 4 minutes 9 seconds
# Start capturing.
# Paused capturing.
# Traceback (most recent call last):
#   File "G:\mecheye_python_samples\area_scan_3d_camera\advanced\capture_periodically.py", line 85, in <module>
#     a.main()
#   File "G:\mecheye_python_samples\area_scan_3d_camera\advanced\capture_periodically.py", line 78, in main
#     self.capture_timed_and_periodically()
#   File "G:\mecheye_python_samples\area_scan_3d_camera\advanced\capture_periodically.py", line 63, in capture_timed_and_periodically
#     time.sleep(capture_period - time_used)
# KeyboardInterrupt
#
# Process finished with exit code -1073741510 (0xC000013A: interrupted by Ctrl+C)
