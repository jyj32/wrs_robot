# With this sample, you can define and register the callback function for monitoring the Stereo_Camera connection status.
# 通过该示例，您可以定义并注册用于监控相机连接状态的回调函数。
import os

# 具体而言，代码定义了一个自定义回调函数，
# 用于处理相机曝光结束事件和相机断开连接事件。


import cv2
import time
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


# 定义回调类 CustomCallback
class CustomCallback(EventCallbackBase):
    def __init__(self):
        super().__init__()

    def process_event(self, eventData):
        print(
            "A Stereo_Camera event has occurred. The event ID is {0}.".format(eventData.event_id))


class RegisterCameraEvent(object):
    def __init__(self):
        self.camera = Camera()

    def capture_depth_map(self):
        r"""
        注意：CAMERA_EVENT_EXPOSURE_END 事件仅在 3D 数据 (Frame3D) 获取完成后发送。
        为确保在发送事件之前已获取 2D 和 3D 数据，请检查以下建议：
        如果使用闪光曝光模式获取 2D 数据，并且 :py:class:FlashAcquisitionMode 参数设置为“Fast”，
        则在调用 capture_2d() 之前调用 capture_3d()。否则，在调用 capture_3d() 之前调用 capture_2d()。
        或者，您可以改为调用 capture_2d_and_3d() 以避免时间问题。

        Note: The CAMERA_EVENT_EXPOSURE_END event is only sent after the acquisition of the 3D data (Frame3D) has completed.
        To ensure both 2D and 3D data have been acquired before the event is sent, check the following recommendations:
        If the flash exposure mode is used for acquiring the 2D data, and the :py:class:FlashAcquisitionMode parameter is set to "Fast",
        call capture_3d() before calling capture_2d(). Otherwise, call capture_2d() before calling capture_3d().
        Alternatively, you can call capture_2d_and_3d() instead to avoid the timing issue.
        """
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))

        depth_map = frame3d.get_depth_map()
        depth_file = os.path.join(save_directory, f"DepthMap.tiff")
        cv2.imwrite(depth_file, depth_map.data())
        print("Capture and save the depth map: {}".format(depth_file))

    def main(self):
        if not find_and_connect(self.camera):
            return

        device_event = CameraEvent()
        callback = CustomCallback()
        print("Register the callback function for Stereo_Camera exposure end event.")
        # CameraEvent 是一个事件类，用于处理相机事件。
        # register_camera_event_callback 方法注册回调函数
        # 这里注册了当相机曝光结束时触发的事件（CAMERA_EVENT_EXPOSURE_END），
        # 并调用 CustomCallback 进行处理。
        show_error(device_event.register_camera_event_callback(
            self.camera, CameraEvent.CAMERA_EVENT_EXPOSURE_END, callback))
        self.capture_depth_map()
        # 在曝光结束事件处理完毕后，注销该回调。
        show_error(device_event.unregister_camera_event_callback(self.camera, CameraEvent.CAMERA_EVENT_EXPOSURE_END))

        # 注册并监听相机断开连接事件（CAMERA_EVENT_DISCONNECTED）。
        # 此事件会在相机断开时触发，回调函数 CustomCallback 会被调用。
        print("Register the callback function for Stereo_Camera disconnection event.")
        show_error(
            device_event.register_camera_event_callback(self.camera, CameraEvent.CAMERA_EVENT_DISCONNECTED, callback))
        time.sleep(20)
        show_error(device_event.unregister_camera_event_callback(self.camera, CameraEvent.CAMERA_EVENT_DISCONNECTED))

        self.camera.disconnect()
        print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = RegisterCameraEvent()
    a.main()

# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\advanced\register_camera_event.py
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
# Register the callback function for Stereo_Camera exposure end event.
# Successfully registered the callback function for the following event: 36878.
# The message channel for delivering the events of this device has been established at 169.254.194.154:48001.
# A Stereo_Camera event has occurred. The event ID is 36878.
# Capture and save the depth map: G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset\DepthMap.tiff
# Register the callback function for Stereo_Camera disconnection event.
#
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
