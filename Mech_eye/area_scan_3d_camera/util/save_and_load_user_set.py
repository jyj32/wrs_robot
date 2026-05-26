# With this sample, you can import and replace all user sets from a JSON file, and save all user sets to a JSON file.
# 通过此示例，您可以从 JSON 文件导入和替换所有用户集，并将所有用户集保存到 JSON 文件中。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class SaveAndLoadUserSet(object):
    def __init__(self):
        self.camera = Camera()
        self.user_set_manager = self.camera.user_set_manager()

    def save_and_load_user_set(self):

        # Obtain the names of all user sets.
        print("All user sets: ", end='')
        error, user_sets = self.user_set_manager.get_all_user_set_names()
        show_error(error)
        for user_set in user_sets:
            print(user_set, end=' ')

        print("Save all user sets to a JSON file.")
        show_error(self.user_set_manager.save_to_file("camera_config.json"))

        print("Import and replace all user sets from a JSON file.")
        show_error(self.user_set_manager.load_from_file("camera_config.json"))

    def main(self):
        if find_and_connect(self.camera):
            self.save_and_load_user_set()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = SaveAndLoadUserSet()
    a.main()

# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\save_and_load_user_set.py
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
# All user sets: default calib Reflective object Translucent object Reflective + unreflective Reflective metal Small carton NewUserSet Save all user sets to a JSON file.
# Import and replace all user sets from a JSON file.
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
#
# '''
