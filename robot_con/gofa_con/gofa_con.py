"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
import time
import numpy as np

from .gofa_arm import GoFaArm
from .gofa_state import GoFaState
from .piecewisepoly import PiecewisePoly
from .gofa_constants import GoFaConstants as GFC


class GoFaArmController:
    """
    A client to control the yumi
    """

    def __init__(self, toggle_debug=False, toggle_monitor_only=False):
        self._toggle_monitor_only = toggle_monitor_only
        if not toggle_monitor_only:
            self.rbtx = GoFaArm(debug=toggle_debug)
            self._is_add_all = True
            self._traj_opt = PiecewisePoly()
        self.sec_rbtx = GoFaArm(debug=toggle_debug, port=GFC.PORTS['states'])

    @property
    def arm(self):
        return self.rbtx if not self._toggle_monitor_only else self.sec_rbtx

    def get_pose(self, component_name, return_conf=False):
        raise NotImplementedError

    def get_jnt_values(self):
        """
        get the joint angles of both arms
        :return: 1x6 array
        author: chen
        """
        return np.deg2rad(self.sec_rbtx.get_state().joints)

    def get_torques(self) -> np.ndarray:
        """
        get the torques of joints
        See 1: https://library.e.abb.com/public/b227fcd260204c4dbeb8a58f8002fe64/Rapid_instructions.pdf
        See 2: https://forums.robotstudio.com/discussion/13247/motor-torque-using-getmotortorque-vs-getjointdata
        Notes: When the GoFa joints lock (idle), the torques are zeros.
        :return: joints
        """
        return np.asarray(self.sec_rbtx.get_torques())

    def get_torques_current(self) -> np.ndarray:
        """
        get the torques of joints
        See 1: https://library.e.abb.com/public/b227fcd260204c4dbeb8a58f8002fe64/Rapid_instructions.pdf
        See 2: https://forums.robotstudio.com/discussion/13247/motor-torque-using-getmotortorque-vs-getjointdata
        Notes: When the GoFa joints lock (idle), the torques are zeros.
        :return: joints
        """
        return np.asarray(self.sec_rbtx.get_torques_current())

    def move_j(self, jnt_vals: np.ndarray, speed_n=100, wait=True):
        """
        机器人关节空间单点运动函数
        输入：关节数组，速度，是否等待完成
        输出：布尔值
        move one arm joints of the gofa
        :param component_name
        :param jnt_vals: 1x6 np.array
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf
        :return: bool
        """
        if self._toggle_monitor_only:   # 安全模式检查，确保不在仅监控模式下执行真实运动
            raise Exception("Toggle off monitor only to enable robot movements")
        assert len(jnt_vals) == GoFaState.NUM_JOINTS
        if speed_n == -1:   # 最大速度
            self.arm.set_speed_max()
        else:   # 设置速度
            speed_data = self.rbtx.get_v(speed_n)
            self.arm.set_speed(speed_data)

        armjnts = np.rad2deg(jnt_vals)  # 弧度转角度
        ajstate = GoFaState(armjnts)    # 创建机器人状态对象
        exec_result = self.arm.movetstate_sgl(ajstate, wait_for_res=wait)     # 移动到目标关节角度
        return exec_result  # 返回布尔值，是否成功移动

    def fk(self, component_name: str, jnt_vals: np.ndarray, return_conf: bool = False) -> tuple:
        raise NotImplementedError

    def ik(self, component_name: str,
           pos: np.ndarray,
           rot: np.ndarray,
           conf: np.ndarray = None,
           ext_axis: float = None) -> np.ndarray or None:
        raise NotImplementedError

    def move_jntspace_path(self, path, speed_n=100, wait=True) -> bool:
        """
        控制机器人沿关节空间路径path运动
        输入：speed_n速度，wait是否等待运动完成，默认True。
        输出：返回一个布尔值表示执行结果
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        """
        if self._toggle_monitor_only:   # 安全模式检查,防止在仅监控模式下执行实际运动
            raise Exception("Toggle off monitor only to enable robot movements")
        statelist = []  # 状态列表
        st = time.time()    # 记录开始时间
        for armjnts in self._traj_opt.interpolate_path(path, num=min(100, int(len(path)))):
            # 路径插值循环，self._traj_opt.interpolate_path对原始路径path进行插值，num插值点数量，取100和len(path)中的较小值
            armjnts = np.rad2deg(armjnts)   # 弧度制转换为角度制
            ajstate = GoFaState(armjnts)    # 将关节角度封装成 GoFaState 对象
            statelist.append(ajstate)   # 添加到状态列表
        et = time.time()    # 记录时间
        print("time calculating sending information", et - st)  # 计算插值时间
        # set the speed of the robot
        if speed_n == -1:   # 机器人设置最大速度
            self.arm.set_speed_max()
        else:   # 机器人设置速度
            speed_data = self.arm.get_v(speed_n)
            self.arm.set_speed(speed_data)
        exec_result = self.arm.movetstate_cont(statelist, is_add_all=self._is_add_all, wait_for_res=wait)   # 连续运动命令，执行一系列状态点
        '''
        statelist：插值后的状态点列表
        is_add_all：是否添加所有状态点
        wait_for_res：是否等待运动完成
        '''
        return exec_result  # 布尔值

    def stop(self):
        if not self._toggle_monitor_only:
            self.rbtx.stop()
        self.sec_rbtx.stop()


if __name__ == "__main__":
    yumi_con = GoFaArmController()
    print(yumi_con.get_jnt_values())
