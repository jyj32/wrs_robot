# With this sample, you can set the parameters specific to laser cameras (the DEEP and LSR series).
# 通过此示例，您可以设置特定于激光相机（DEEP和LSR系列）的参数。

# 这段代码是一个示例，展示了如何通过 Mech-Eye SDK 设置激光相机（DEEP 和 LSR 系列）的相关参数。
# 具体来说，它演示了如何调整激光扫描的不同设置，以优化拍摄效果。


from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


# SetParametersOfLaserCameras 是一个类，负责设置激光相机的各种参数。
class SetParametersOfLaserCameras(object):
    def __init__(self):
        self.camera = Camera()

    #  设置激光分区数(set_laser_partition_count)
    def set_laser_partition_count(self):
        current_user_set = self.camera.current_user_set()
        error, laser_partition_count = current_user_set.get_int_value(LaserFramePartitionCount.name)
        show_error(error)
        print("Old frame partition count: {}".format(laser_partition_count))

        # Set the laser scan partition count. If the set value is greater than 1, the scan of the entire FOV
        # will be partitioned into multiple parts. It is recommended to use multiple parts for
        # extremely dark objects.
        error = current_user_set.set_int_value(LaserFramePartitionCount.name, 2)
        show_error(error)
        error, laser_partition_count = current_user_set.get_int_value(LaserFramePartitionCount.name)
        show_error(error)
        print("New frame partition count: {}".format(laser_partition_count))

    # 设置激光扫描范围(set_laser_frame_range)
    def set_laser_frame_range(self):
        current_user_set = self.camera.current_user_set()
        error, laser_frame_range = current_user_set.get_range_value(LaserFrameRange.name)
        show_error(error)
        print("Old frame range: {} to {}.".format(laser_frame_range.min, laser_frame_range.max))

        # Set the laser scan range. The entire projector FOV is from 0 to 100.
        laser_frame_range.min = 20
        laser_frame_range.max = 80
        error = current_user_set.set_range_value(LaserFrameRange.name, laser_frame_range)
        show_error(error)
        error, laser_frame_range = current_user_set.get_range_value(LaserFrameRange.name)
        show_error(error)
        print("New frame range: {0} to {1}.".format(laser_frame_range.min, laser_frame_range.max))

    # 设置激光条纹编码模式 (set_laser_fringe_coding_mode)
    # set_laser_fringe_coding_mode 方法设置激光条纹的编码模式。
    # 条纹编码模式控制了结构光的模式。
    # 可以选择 Fast 或 Accurate 模式：
    # Fast 模式：提高捕获速度，但深度数据精度较低。
    # Accurate 模式：提供更高的深度数据精度，但速度较慢。

    def set_laser_fringe_coding_mode(self):
        current_user_set = self.camera.current_user_set()
        error, laser_fringe_coding_mode = current_user_set.get_enum_value_string(LaserFringeCodingMode.name)
        print("Old fringe coding mode: {}.".format(laser_fringe_coding_mode))

        # Set the "Fringe Coding Mode" parameter, which controls the pattern of the structured light.
        # The "Fast" mode enhances the capture speed but provides lower depth data accuracy.
        # The "Accurate" mode provides better depth data accuracy but reduces the capture speed.
        # 设置“条纹编码模式”参数，该参数控制结构光的模式。
        # “快速”模式可提高捕获速度，但深度数据精度较低。
        # “准确”模式可提供更好的深度数据精度，但会降低捕获速度。
        error = current_user_set.set_enum_value(LaserFringeCodingMode.name, LaserFringeCodingMode.Value_Accurate)
        show_error(error)
        error, laser_fringe_coding_mode = current_user_set.get_enum_value_string(LaserFringeCodingMode.name)
        show_error(error)
        print("New fringe coding mode: {}.".format(laser_fringe_coding_mode))

    # set_laser_power_level 方法设置激光投影的功率级别。
    # 功率级别影响投影的强度，可以调节投影的亮度。
    # 在这里将功率设置为 80，即占最大输出功率的 80 %。
    def set_laser_power_level(self):
        current_user_set = self.camera.current_user_set()
        error, laser_power_level = current_user_set.get_int_value(LaserPowerLevel.name)
        show_error(error)
        print("Old power level: {}".format(laser_power_level))

        # Set the "Laser Power" parameter,
        # which is the output power of the projector as a percentage of the maximum output power. This affects the
        # intensity of the projected structured light.
        # 设置“激光功率”参数，即投影仪的输出功率占最大输出功率的百分比。这会影响投影结构光的强度。

        laser_power_level = 80
        error = current_user_set.set_int_value(LaserPowerLevel.name, laser_power_level)
        show_error(error)
        error, laser_power_level = current_user_set.get_int_value(LaserPowerLevel.name)
        show_error(error)
        print("New power level: {}".format(laser_power_level))

    def main(self):
        if find_and_connect(self.camera):
            self.set_laser_power_level()
            self.set_laser_fringe_coding_mode()
            self.set_laser_frame_range()
            self.set_laser_partition_count()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = SetParametersOfLaserCameras()
    a.main()
