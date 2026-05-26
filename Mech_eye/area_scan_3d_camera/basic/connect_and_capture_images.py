# With this sample, you can connect to a Stereo_Camera and obtain the 2D image, depth map, and point cloud data.
# 通过该示例，您可以连接摄像头，获取二维图像、深度图和点云数据。


from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *


class ConnectAndCaptureImages(object):
    def __init__(self):
        self.camera = Camera()

    def connect_and_capture(self):

        # Obtain the 2D image resolution and the depth map resolution of the Stereo_Camera.
        resolution = CameraResolutions()
        show_error(self.camera.get_camera_resolutions(resolution))
        print_camera_resolution(resolution)

        # Obtain the 2D image.
        frame2d = Frame2D()
        show_error(self.camera.capture_2d(frame2d))
        row, col = 222, 222
        color_map = frame2d.get_color_image()
        print("The size of the 2D image is {} (width) * {} (height).".format(
            color_map.width(), color_map.height()))
        rgb = color_map[row * color_map.width() + col]
        print("The RGB values of the pixel at ({},{}) is R:{},G:{},B{}\n".
              format(row, col, rgb.b, rgb.g, rgb.r))

        if not confirm_capture_3d():
            return

        # Obtain the depth map.
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))
        depth_map = frame3d.get_depth_map()
        print("The size of the depth map is {} (width) * {} (height).".format(
            depth_map.width(), depth_map.height()))
        depth = depth_map[row * depth_map.width() + col]
        print("The depth value of the pixel at ({},{}) is depth :{}mm\n".
              format(row, col, depth.z))

        # Obtain the point cloud.
        point_cloud = frame3d.get_untextured_point_cloud()
        print("The size of the point cloud is {} (width) * {} (height).".format(
            point_cloud.width(), point_cloud.height()))
        point_xyz = point_cloud[row * depth_map.width() + col]
        print("The coordinates of the point corresponding to the pixel at ({},{}) is X: {}mm , Y: {}mm, Z: {}mm\n".
              format(row, col, point_xyz.x, point_xyz.y, point_xyz.z))

    def main(self):
        # List all available cameras and connect to a Stereo_Camera by the displayed index.
        if find_and_connect(self.camera):
            self.connect_and_capture()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = ConnectAndCaptureImages()
    a.main()

# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\basic\connect_and_capture_images.py
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
# Texture Map size : (width : 1920, height : 1200).
# Depth Map size : (width : 1920, height: 1200).
# The size of the 2D image is 1920 (width) * 1200 (height).
# The RGB values of the pixel at (222,222) is R:58,G:58,B58
#
# Do you want the Stereo_Camera to capture 3D image ? Please input y/n to confirm:
# y
# The size of the depth map is 1920 (width) * 1200 (height).
# The depth value of the pixel at (222,222) is depth :1230.2919921875mm
#
# The size of the point cloud is 1920 (width) * 1200 (height).
# The coordinates of the point corresponding to the pixel at (222,222) is X: -374.20965576171875mm , Y: -208.33644104003906mm, Z: 1230.2919921875mm
#
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
#
# '''
