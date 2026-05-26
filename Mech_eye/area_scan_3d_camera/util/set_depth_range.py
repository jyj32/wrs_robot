# With this sample program, you can set the range of depth values to be retained by a Stereo_Camera.
# 通过此示例程序，您可以设置相机保留的深度值范围。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class SetDepthRange(object):
    def __init__(self):
        self.camera = Camera()

    def set_depth_range(self):
        user_set = self.camera.current_user_set()
        # Set the range of depth values to 100–1000 mm.
        depth_range = RangeInt(100, 1000)
        show_error(user_set.set_range_value(
            Scanning3DDepthRange.name, depth_range))
        print("\n3D scanning depth lower limit : {} mm,".format(depth_range.min),
              "depth upper limit : {} mm\n".format(depth_range.max))

    def main(self):
        if find_and_connect(self.camera):
            self.set_depth_range()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = SetDepthRange()
    a.main()


# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\set_depth_range.py
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
#
# 3D scanning depth lower limit : 100 mm, depth upper limit : 1000 mm
#
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
#
# '''