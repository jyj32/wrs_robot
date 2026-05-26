# With this sample, you can set the parameters specific to the UHP series.
# 通过此示例，您可以设置特定于UHP系列的参数。
#
# 这段代码的主要目的是设置特定于 UHP 系列相机的参数，具体是设置相机的捕获模式为 "Merge" 模式。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


class SetParametersOfUHPCameras(object):
    def __init__(self):
        self.camera = Camera()

    def set_uhp_capture_mode(self):
        current_user_set = self.camera.current_user_set()
        error, uhp_settings = current_user_set.get_enum_value_string(
            UhpCaptureMode.name)
        show_error(error)
        print("Old capture mode: {}.".format(uhp_settings))

        # set the capture mode to "Merge"
        error = current_user_set.set_enum_value(
            UhpCaptureMode.name, UhpCaptureMode.Value_Merge)
        show_error(error)
        error, uhp_settings = current_user_set.get_enum_value_string(
            UhpCaptureMode.name)
        show_error(error)
        print("New capture mode: {}.".format(uhp_settings))

    def main(self):
        if find_and_connect(self.camera):
            self.set_uhp_capture_mode()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = SetParametersOfUHPCameras()
    a.main()
