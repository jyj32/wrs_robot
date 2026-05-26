# With this sample, you can manage user sets, such as obtaining the names of all user sets, adding a user set, switching the user set, and saving parameter settings to the user set.
# 通过该样例，你可以实现对用户集的管理，比如获取所有用户集的名称、添加用户集、切换用户集、保存参数设置到用户集等。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class ManageUserSets(object):
    def __init__(self):
        self.camera = Camera()

    def manage_user_set(self):
        user_set_manager = self.camera.user_set_manager()

        # Obtain the names of all user sets.
        user_set_manager.delete_user_set("NewUserSet")
        print("All user sets: ", end='')
        error, user_sets = user_set_manager.get_all_user_set_names()
        show_error(error)
        for user_set in user_sets:
            print(user_set, end=' ')

        # Obtain the name of the currently selected user set.
        error, name = user_set_manager.current_user_set().get_name()
        show_error(error)
        print("\nCurrent user set: " + str(name))

        # Add a user set.
        new_set = "NewUserSet"
        show_error(user_set_manager.add_user_set(new_set))
        print("Add a new user set : \"{}\".".format(new_set))

        # Select a user set by its name.
        show_error(user_set_manager.select_user_set(new_set))
        print("select \"{}\" as the current user set.".format(new_set))

        show_error(user_set_manager.current_user_set(
        ).save_all_parameters_to_device())
        print("Save all parameters to current user set.")

    def main(self):
        if find_and_connect(self.camera):
            self.manage_user_set()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = ManageUserSets()
    a.main()


# '''
# D:\Anaconda3\envs\Mech-Eye\python.exe G:\mecheye_python_samples\area_scan_3d_camera\util\manage_user_sets.py
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
# All user sets: default calib Reflective object Translucent object Reflective + unreflective Reflective metal Small carton
# Current user set: calib
# Add a new user set : "NewUserSet".
# select "NewUserSet" as the current user set.
# Save all parameters to current user set.
# Disconnected from the Stereo_Camera successfully.
#
# Process finished with exit code 0
#
# '''