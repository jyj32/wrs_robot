import math
import time
import numpy as np

# Assuming these custom libraries from the original project are available
# import motion.trajectory.polynomial_wrsold as pwp
from basis import robot_math as rm
# from drivers.devices.dh.ag95 import Ag95Driver

# Import the new UR RTDE library components
import rtde_control     # ur机器人库
import rtde_receive     # ur机器人库
import rtde_io
import drivers.urx.ur_robot as urrobot
import robot_con.ur.program_builder_dh as pb
from pathlib import Path
import motion.trajectory.piecewisepoly_toppra as pwp

# 机器人配置文件
class UR5Ag95X_RTDE(object):
    """
    A refactored version of the UR5 control class using the ur_rtde library.

    author: weiwei (original), Gemini (refactor)
    date: 20250704
    """

    def __init__(self,
                 robot_ip='192.168.125.30',
                 gp_port='com5'):
        """
        Initializes the robot arm and gripper using the ur_rtde library.

        :param robot_ip: The IP address of the UR5 robot.
        :param gp_port: The COM port for the Ag95 gripper.
        """
        # --- Setup Arm using ur_rtde ---
        self.rtde_frequency = 1000.0
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip, self.rtde_frequency)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, self.rtde_frequency)
        self._rtde_io = rtde_io.RTDEIOInterface(robot_ip, self.rtde_frequency)
        # self._arm  = urrobot.URRobot(robot_ip)    # 工厂里要注释掉
        # Set default TCP and payload
        # self._rtde_c.setTCP((0, 0, 0, 0, 0, 0))
        # Original payload was 1.28kg. Assuming Center of Gravity is at the flange.
        self._rtde_c.setPayload(1.28, (0, 0, 0))
        self.pb = pb.ProgramBuilder()

        # --- Setup Hand ---
        # This part remains unchanged as it's for a separate device
        # self._hnd = Ag95Driver(port=gp_port)

        # --- Setup Trajectory Planner ---
        # This part remains unchanged
        # self.trajt = pwp.TrajPoly(method='quintic')

        print(f"Successfully connected to UR7 at {robot_ip} and gripper at {gp_port}.")

    def check_rtdec_is_connected(self): # 检查控制是否连接，掉线的话就重连
        """Check if RTDE receive connection is alive."""
        for i in range(3):
            if self._rtde_c.isConnected():  # 已连接
                print("rtde_c已连接")
                return True
            # 掉线，尝试重连
            print(f"RTDE_c连接掉线，第{i + 1}次重连...")
            try:
                self._rtde_c.reconnect()
            except Exception as e:
                print(f"重连异常: {e}")
        else:
            print("rtdec三次重连仍失败")
            return False

    def check_rtder_is_connected(self): # 返回控制是否连接
        """Check if RTDE receive connection is alive."""
        for i in range(3):
            if self._rtde_r.isConnected():  # 已连接
                print("rtde_r已连接")
                return True
            # 掉线，尝试重连
            print(f"RTDE_r连接掉线，第{i + 1}次重连...")
            try:
                self._rtde_r.reconnect()
            except Exception as e:
                print(f"重连异常: {e}")
        else:
            print("rtder三次重连仍失败")
            return False

    def open_gripper_dh(self, speed=100, force=100):
        SCRIPT_PATH = Path(r"E:\py_project\wrsrobot\wrs_shu\drivers\devices\dh\ac.script")
        prog = self.pb.get_str_from_file(SCRIPT_PATH)
        prog = prog.replace("program_replace_speed",   f"dh_pgc_set_speed(1,{speed})")  # 用speed替换速度
        prog = prog.replace("program_replace_force",   f"dh_pgc_set_force(1,{force})")  # 用force替换力
        prog = prog.replace("program_replace_command", f'dh_pgc_set_position(1,100)')
        # print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()
        self._rtde_c.reconnect()

    def close_gripper_dh(self, speed=100, force=100):
        SCRIPT_PATH = Path(r"E:\py_project\wrsrobot\wrs_shu\drivers\devices\dh\ac.script")
        prog = self.pb.get_str_from_file(SCRIPT_PATH)
        prog = prog.replace("program_replace_speed",   f"dh_pgc_set_speed(1,{speed})")  # 用speed替换速度
        prog = prog.replace("program_replace_force",   f"dh_pgc_set_force(1,{force})")  # 用force替换力
        prog = prog.replace("program_replace_command", f'dh_pgc_set_position(1,0)')
        # print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()   # 断线
        self._rtde_c.reconnect()    # 重连

    def close_to_dh(self, width, speed=100, force=100):
        SCRIPT_PATH = Path(r"E:\py_project\wrsrobot\wrs_shu\drivers\devices\dh\ac.script")
        prog = self.pb.get_str_from_file(SCRIPT_PATH)
        prog = prog.replace("program_replace_speed",   f"dh_pgc_set_speed(1,{speed})")
        prog = prog.replace("program_replace_force",   f"dh_pgc_set_force(1,{force})")
        prog = prog.replace("program_replace_command", f'dh_pgc_set_position(1,{width})')
        # print(prog)
        self._arm.send_program(prog)
        self._rtde_c.disconnect()
        self._rtde_c.reconnect()
        # self._arm.secmon.close()

    def get_dh_wide(self):
        SCRIPT_PATH = Path(r"E:\py_project\wrsrobot\wrs_shu\drivers\devices\dh\ac.script")
        prog = self.pb.get_str_from_file(SCRIPT_PATH)
        prog = prog.replace("dh_pgc_get_position",   f"(1)")
        # print(prog)
        self._arm.send_program(prog)
        return
        # self._arm.secmon.close()

    @property
    def rtde_c(self):
        """Read-only property for the RTDE control interface."""
        return self._rtde_c

    @property
    def rtde_r(self):
        """Read-only property for the RTDE receive interface."""
        return self._rtde_r

    @property
    def hnd(self):
        """Read-only property for the gripper driver."""
        return self._hnd

    # def open_gripper(self):
    #     """Opens the Ag95 gripper."""
    #     self.hnd.open_g()
    #     print("Gripper opened.")
    #
    # def slow_open_gripper(self):
    #     self.hnd.set_speed(80)
    #     self.hnd.open_g()
    #     self.hnd.set_speed(100)
    #
    # def close_gripper(self):
    #     """Closes the Ag95 gripper."""
    #     self.hnd.close_g()
    #     print("Gripper closed.")

    def move_jnts(self, jnt_values, vel=1.0, acc=1.0, wait=True):
        """
        Moves the robot to a target joint configuration.

        :param jnt_values: A list of 6 joint angles in radians.
        :param vel: Joint velocity in rad/s.速度范围0-3.14
        :param acc: Joint acceleration in rad/s^2.
        :param wait: If True, blocks until the movement is complete.
        """
        # The 'asynchronous' parameter in moveJ is the opposite of 'wait'
        self.check_rtdec_is_connected() # 检查_rtde_c是否连接
        self._rtde_c.moveJ(jnt_values, vel, acc, not wait)

    def moveL(self, pose, vel=3.0, acc=3.0, wait=True): # 走直线，姿态也可以旋转
        """
        Move to position (linear in tool-space)
        Parameter ``pose``:
            target pose 目标位姿
        Parameter ``speed``:
            tool speed [m/s]
        Parameter ``acceleration``:
            tool acceleration [m/s^2]
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
        # The 'asynchronous' parameter in moveJ is the opposite of 'wait'
        # 检查rtde连接
        self.check_rtdec_is_connected() # 检查_rtde_c是否连接
        if acc > 3.14 : # 可能最大是3.14，实验一下
            acc = 3.14
        self._rtde_c.moveL(pose, vel, acc, not wait)

    def servoL(self, pose, time=0.1, lookahead_time=0.1, gain=300, wait=True):
        """
        伺服运动到目标位姿（线性运动）

        Args:
            pose: 目标位姿 [x,y,z,rx,ry,rz]
            time: 控制时间 [秒]，函数会阻塞这段时间
            lookahead_time: 前瞻时间 [秒]，范围 [0.03, 0.2]
            gain: 位置比例增益，范围 [100, 2000]
            wait: 是否等待控制完成，默认 True
                  True: 函数会阻塞 time 秒
                  False: 立即返回，但伺服控制仍在后台运行

        Returns:
            bool: 伺服命令是否成功发送
        """
        # 参数验证（可选）
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise ValueError("pose must be a list/tuple of 6 floats")

        # speed 和 acceleration 参数虽然未使用，但必须传入 float 值
        unused = 0.0

        # 调用底层函数
        result = self._rtde_c.servoL(pose, unused, unused, time, lookahead_time, gain)

        # 如果需要等待，阻塞 time 秒
        if wait:
            import time as _time
            _time.sleep(time)

        return result

    def regulate_jnts_pmpi(self):
        """
        Moves all joints to their equivalent angle within the [-pi, pi] range.
        Useful for resetting joint configurations after multiple rotations.
        """
        current_jnts = self.get_jnt_values()
        regulated_jnts = rm.regulate_angle(-math.pi, math.pi, current_jnts)
        print("Regulating joints to [-pi, pi] range.")
        self.move_jnts(regulated_jnts)

    def move_jntspace_path(self, path, interval_time=1.0, control_frequency=.002,
                           vel=0.5, acc=0.8,
                           speed_gain=300,
                           blend=0.0, toppra_vels = [0.1,0.1,0.1,0.1,0.1,0.1], toppra_accs = [1,1,1,1,1,1]):
        """
        Executes a trajectory defined by a list of joint-space waypoints.

        :param path: A list of 1x6 joint configurations (in radians).
        :param interval_time: Time interval for trajectory interpolation.
        :param control_frequency: Control frequency for interpolation.
        :param vel: Velocity for each point in the trajectory.
        :param acc: Acceleration for each point in the trajectory.
        :param blend: Blend radius for smoothing corners between points.
        """
        # Interpolate the path using the provided trajectory planner
        # interpolated_confs, _, _ = self.trajt.piecewise_interpolation(path, control_frequency, interval_time)
        #
        # # Format the trajectory for the ur_rtde moveJ command
        # # Each point needs: [q1...q6,]
        # rtde_path = interpolated_confs
        # print(f"Executing trajectory with {len(rtde_path)} points...")
        # # Send the entire trajectory to the robot. This call is synchronous and waits for completion.
        self.check_rtdec_is_connected()  # 检查_rtde_c是否连接

        velocity = vel
        acceleration = acc
        dt = 1.0 / self.rtde_frequency  # 2ms
        lookahead_time = 0.1   # 默认为0.1
        gain = speed_gain
        # for q_joint in rtde_path:
        #     t_start = self._rtde_c.initPeriod()
        #     self._rtde_c.servoJ(q_joint, velocity, acceleration, dt, lookahead_time, gain)
        #     self._rtde_c.waitPeriod(t_start)
        # self._rtde_c.servoStop()  # Stop the servo motion
        # print("Trajectory execution complete.")

        control_frequency = control_frequency
        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                            control_frequency=control_frequency,
                                                            max_vels=toppra_vels,
                                                            max_accs=toppra_accs,
                                                            toggle_debug=False)
        interpolated_path = interpolated_path[1:]
        for q_joint in interpolated_path:
            t_start = self._rtde_c.initPeriod()
            self._rtde_c.servoJ(q_joint, velocity, acceleration, dt, lookahead_time, gain)
            self._rtde_c.waitPeriod(t_start)

        self._rtde_c.servoStop()  # 关键！退出伺服模式

        return

    def get_jnt_values(self):
        """
        Returns the current joint angles of the robot in radians.
        :return: A numpy array of 6 joint angles.
        """
        return np.asarray(self._rtde_r.getActualQ())

    def get_pose(self):
        """
        Returns the current TCP pose of the robot.

        :return: A tuple (pos, rot) where:
                 - pos is a 1x3 numpy array for [x, y, z].
                 - rot is a 3x3 numpy array representing the rotation matrix.
        """
        tcp_pose_vec = self._rtde_r.getActualTCPPose()
        pos = np.asarray(tcp_pose_vec[:3])
        # Convert the rotation vector to a 3x3 rotation matrix
        rot = rm.rotmat_from_euler(tcp_pose_vec[3], tcp_pose_vec[4], tcp_pose_vec[5])
        return pos, rot


    def getTCPForce_z(self):
        Fz = self._rtde_r.getActualTCPForce()[2]
        return Fz

    def io(self, ip, level):
        self._rtde_io.setStandardDigitalOut(ip, level)

    def green_lamp(self): # 绿灯亮
        self.io(1,0)
        time.sleep(1)
        self.io(2, 1)
        time.sleep(1)

    def red_lamp(self): # 红灯亮
        self.io(2, 0)
        time.sleep(1)
        self.io(1,1)
        time.sleep(1)


if __name__ == '__main__':
    rbt_r = UR5Ag95X_RTDE(
        robot_ip='192.168.125.30',
        gp_port='COM5'
    )
    print(rbt_r.get_jnt_values())   # 获得关节角度
    # while True:
    #     print(rbt_r.getTCPForce_z())
    #     time.sleep(10)


    # import socket
    # # This example requires the user's custom libraries to run.
    # # The structure of the main block is preserved.
    # # import visualization.panda.world as wd
    # # base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    #
    # task_frame = [0, 0, 0, 0, 0, 0]
    # selection_vector = [0, 0, 1, 1, 1, 0]
    # # wrench_down = [0, 0, -10, 0, 0, 0]
    # wrench_down = [0, 0, -10, 0, 0, 0]
    # force_type = 2
    # limits = [2, 2, 2, 1, 1, 1]
    # # dt = 1.0 / 500  # 2ms
    # # joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]
    #
    # robot_ip = "192.168.125.30"
    # u5rag95_x1 = UR5Ag95X_RTDE(robot_ip=robot_ip, gp_port='COM5')
    # # u5rag95_x2 = UR5Ag95X_RTDE(robot_ip=robot_ip, gp_port='COM5')
    # u5rag95_x1.open_gripper_dh()
    # u5rag95_x1.rtde_c.disconnect()
    # u5rag95_x1.rtde_c.reconnect()
    # u5rag95_x1.rtde_c.zeroFtSensor()
    # # u5rag95_x1.rtde_c.forceMode(task_frame, selection_vector, wrench_down, force_type, limits)
    # # u5rag95_x1.rtde_c.forceMode(task_frame = [1.0,0.0,0.0,0.0,0.0,0.0], selection_vector=[1, 0, 0, 0, 0, 0], wrench=[5.0,0.0,0.0,0.0,0.0,0.0], type=2, limits = [.1,.1,.1,.785,.785,1.57])
    #
    # while u5rag95_x1.rtde_r.getActualTCPForce()[2]<9.0:
    #     u5rag95_x1.rtde_c.forceMode(task_frame, selection_vector, wrench_down, force_type, limits)
    #     u5rag95_x1._rtde_c.waitPeriod(u5rag95_x1._rtde_c.initPeriod())
    # u5rag95_x1.close_gripper_dh()
    # u5rag95_x1.rtde_c.disconnect()
    # u5rag95_x1.rtde_c.reconnect()
    # u5rag95_x1.rtde_c.zeroFtSensor()
    # u5rag95_x1.rtde_c.forceMode(task_frame, selection_vector, [0, 0, 10, 0, 0, 0], force_type, limits)

    # u5rag95_x1.forceModeStop()
    # while u5rag95_x1.rtde_r.getActualTCPForce()[2] < 9.0:


    # u5rag95_x1.rtde_c.forceModeStop()
    # u5rag95_x1.open_gripper_dh()
    # u5rag95_x1.close_gripper_dh()
    # u5rag95_x1.close_to_dh(100,00,100)
    # u5rag95_x1.close_to_dh(50, 00, 100)
    # u5rag95_x1.close_to_dh(0, 00, 100)
    # u5rag95_x1.open_gripper_dh()
    # u5rag95_x1.close_gripper_dh(100, 100, 100)
    # retries = 30
    # connect_timeout = 2.0
    # for attempt in range(1, retries + 1):
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.settimeout(connect_timeout)
    #     try:
    #         s.connect((robot_ip, 30001))
    #         s.sendall(text.encode("utf-8"))
    #         s.close()
    #         print(f"[send_urscript] success on attempt {attempt}")
    #     except OSError as e:
    #         print(f"[send_urscript] attempt {attempt} failed: {e}")
    #         try:
    #             s.close()
    #         except Exception:
    #             pass
    #         if attempt < retries:
    #             time.sleep(0.005)
    # print("=== check ===")



    #
    # print(u5rag95_x1.get_jnt_values())
    # command1 = (
    #     "def main(): \n"
    #     "\tmovej([1.32740247, -1.368692  ,  1.25876171, -1.98938002 , 3.68458843  , 3.30023894])\n"
    #     "\ta = get_actual_joint_positions() \n"
    #     "\ttextmsg(\"11\")\n"
    #     "end\n"
    #     "main()\n"
    # )
    #
    # command2 = (
    #     "def main(): \n"
    #     "\ttextmsg(\"22\")\n"
    #     "\tdh_pgc_scan()\n"
    #     "\tsleep(1)\n"
    #     "\tdh_pgc_relate_device(1,\"TCI\",2)\n"
    #     "\tsleep(1)\n"
    #     "\tdh_pgc_connect(1)\n"
    #     "\tsleep(1)\n"
    #     "\tdh_pgc_set_activate(1)\n"
    #     "end\n"
    #     "main()\n"
    # )
    # print(command2)
    # send_urscript_via_30002('192.168.125.30', command2)
    # print(u5rag95_x1.get_jnt_values())
    # # u5rag95_x1.rtde_c.sendCustomScript(command1)
    # # u5rag95_x1.rtde_c.stopScript()
    # print("check")


    # if __name__ == '__main__':
    #     # 创建机器人实例
    #     rbt_r = UR5Ag95X_RTDE(
    #         robot_ip='192.168.125.30',
    #         gp_port='COM5'
    #     )
    #
    #     try:
    #         # 测试力控制
    #         rbt_r.force_control_contact_stack()
    #         print("力控制测试完成")
    #     except Exception as e:
    #         print(f"测试过程中出错: {e}")
    #         import traceback
    #
    #         traceback.print_exc()
    #     finally:
    #         # 断开连接
    #         rbt_r.disconnect()
