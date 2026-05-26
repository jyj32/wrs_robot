import time
import robot_con.ur.ur7_dh50_rtde as ur7con
import numpy as np

class ForceControl:
    def __init__(self,robot_real: ur7con.UR5Ag95X_RTDE =None): # -z轴方向力控
        self.rbt_r = robot_real
        self.task_frame = [0, 0, 0, 0, 0, 0]
        self.selection_vector = [0, 0, 1, 0, 0, 0]  # 力方向：z轴方向
        self.wrench_down = [0, 0, -10, 0, 0, 0]  # 力大小：-10
        self.force_type = 2
        self.limits = [20, 20, 20, 1, 1, 1]

    def move(self, timeout):    # 下移抓取或放置
        # 初始化变量
        previous_pose = None
        previous_time = None
        period = 0.1  # 检测时间周期
        POSITION_THRESHOLD = 0.001  # 停止阈值0.001m
        stop = 0  # 位移停止标志
        if not self.rbt_r.check_rtdec_is_connected():
            print("rtdec重连失败，力控程序失败")
            return
        # 执行力控制
        self.rbt_r.rtde_c.forceMode(self.task_frame, self.selection_vector, self.wrench_down, self.force_type, self.limits)  # 施加力
        time.sleep(0.3) # 固定力控0.3s
        start_time = time.time()
        while True:
            self.rbt_r.rtde_c.waitPeriod(self.rbt_r.rtde_c.initPeriod())  # 等待一个控制时间
            current_time = time.time()  # 当前时间
            if previous_time is None or current_time - previous_time >= period:
                # 第一次运行时初始化
                if previous_time is None:
                    previous_pose = self.rbt_r.rtde_r.getActualTCPPose()
                    previous_time = current_time
                    continue
                # 检查rtde_r是否连接
                if not self.rbt_r.check_rtder_is_connected():
                    print("rtde_r重连失败,退出力控")
                    self.rbt_r.rtde_c.forceMode(self.task_frame, self.selection_vector, [0] * 6, self.force_type,self.limits)
                    self.rbt_r.rtde_c.forceModeStop()  # 退出力控
                    break
                # 获取当前位姿
                current_pose = self.rbt_r.rtde_r.getActualTCPPose()
                # print(f"当前位置：{current_pos}")
                # 计算三维空间位置变化
                pose_change = sum((current_pose[i] - previous_pose[i]) ** 2 for i in range(3)) ** 0.5
                print(f"力控位置检查间隔: {period:.2f}秒, 位置变化: {pose_change:.6f}m")
                if 0 < pose_change < POSITION_THRESHOLD:
                    stop += 1
                    if stop >= 3:
                        self.rbt_r.rtde_c.forceMode(self.task_frame, self.selection_vector, [0]*6, self.force_type,self.limits)
                        self.rbt_r.rtde_c.forceModeStop()  # 退出力控
                        print(f"到达目标点前{period * 3}秒内位置变化小于{POSITION_THRESHOLD * 3}m")
                        break
                else:  # 清零
                    stop = 0
                # 更新参考位置和时间戳
                previous_pose = current_pose
                previous_time = current_time
            # 检查超时
            if current_time - start_time > timeout:
                print("力控到时，退出力控")
                self.rbt_r.rtde_c.forceMode(self.task_frame, self.selection_vector, [0]*6, self.force_type, self.limits)
                self.rbt_r.rtde_c.forceModeStop()  # 退出力控
                break


if __name__ == "__main__":
    rbt_r = ur7con.UR5Ag95X_RTDE(
        robot_ip='192.168.125.30',  # 机器人IP
        gp_port='COM5'  # 夹爪端口
    )
    ForceControl = ForceControl(rbt_r)
    ForceControl.move(0.7)
