import numpy as np
import basis.robot_math as rm
import time
from config import CONFIG_U1,CONFIG_U625    # 固定参数
from config import CONFIG_U625_cc_xipan,CONFIG_U1_cc_xipan
import robot_con.ur.ur7_dh50_rtde as ur7con
import drivers.devices.dh.ag145 as gripper  # dh手爪
import force_control_move as fcm    # 力控程序
from yanpu_ur8.force_control_move import ForceControl
import sys

class Vacuum_Clean: # 吸盘程序
    def __init__(self, rbt_r, Force_Control): # 初始化
        self.rbt_r = rbt_r
        self.Force_Control = Force_Control  # 力控

    def vacuum_clean_by_layer_U1_1(self, layer_num, fast_v, fast_a, slow_v, slow_a):
        # self.rbt_r.moveL(CONFIG_U1['grasp']['wait_conf'], fast_v, fast_a)
        # 手爪力设为100

        # 设置io信号，不吸气
        self.rbt_r.io(0, 0)
        print('开始')
        if layer_num == 1 or layer_num == 2:
            # 从等待点到点1，接近
            print('//////')
            print('开始')
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean1']['path1'][0], vel=slow_v, acc=slow_a)
            # 到抓取低点
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path1'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 到抓取点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose1'],slow_v,slow_a)

        elif layer_num == 3:
            sys.exit(0)
        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)

    """
    根据层数执行吸盘清理操作

    参数:
        layer_num: 当前层数 (1, 2, 3)
        fast_v: 快速运动速度
        fast_a: 快速运动加速度
        slow_v: 慢速运动速度
        slow_a: 慢速运动加速度
    """

    # 根据不同层数设置不同的高度补偿

    def vacuum_clean_by_layer_U1_2(self, layer_num, fast_v, fast_a, slow_v, slow_a):
        print('开始')
        if layer_num == 1:
            # moveL回到抓取低点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose0'], slow_v, slow_a)
            # 执行吸盘操作，从抓取低点到纸板上方低点
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean1']['path2'][0],slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path2'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 1)
            time.sleep(0.5)
            # moveL到吸取点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose5'], slow_v, slow_a)
            # 力控
            self.Force_Control.move(2)
            time.sleep(2)
            # 从点2到点1，离开
            # moveL到箱子低点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose4'],slow_v,slow_a)
            # 丢弃
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean1']['path4'][0],fast_v,fast_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path4'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 0)
            # 回到抓取低点
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean1']['path5'][0],slow_v,slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path5'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 回到抓取点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose1'], slow_v, slow_a)

        elif layer_num == 2:
            # moveL回到抓取低点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose0'], slow_v, slow_a)
            # 执行吸盘操作，从抓取低点到纸板上方低点
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean2']['path2'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean2']['path2'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 1)
            time.sleep(0.5)
            # moveL到吸取点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean2']['pose5'], slow_v, slow_a)
            # 力控
            self.Force_Control.move(2)
            time.sleep(2)
            # 从点2到点1，离开
            # moveL到箱子低点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean2']['pose4'], slow_v, slow_a)
            # 丢弃
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean2']['path4'][0], fast_v, fast_a)
            # 从点2到点1，离开
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean2']['path4'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 0)
            # 回到抓取低点
            self.rbt_r.move_jnts(CONFIG_U1_cc_xipan['vacuum_clean1']['path5'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path5'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 回到抓取点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose1'], slow_v, slow_a)

        elif layer_num == 3:
            height_offset = -0.04  # 第三层下降4cm
            print(f"执行第{layer_num}层吸盘清理")
            time.sleep(10)
        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)
            height_offset = 0.0

    def vacuum_clean_by_layer_U1_3(self, layer_num, fast_v, fast_a, slow_v, slow_a):
        print('回到等待点')
        if layer_num == 1 or layer_num == 2:
            # 回到抓取低点
            self.rbt_r.moveL(CONFIG_U1_cc_xipan['vacuum_clean1']['pose0'],slow_v,slow_a)
            # 回到等待点
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U1_cc_xipan['vacuum_clean1']['path1'][::-1],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.move_jnts(CONFIG_U1['grasp']['wait_conf'], vel=slow_v, acc=slow_a)

        elif layer_num == 3:
            height_offset = -0.04  # 第三层下降4cm
            print(f"执行第{layer_num}层吸盘清理")
            time.sleep(10)
        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)
            height_offset = 0.0

    def vacuum_clean_by_layer_U625_1(self, layer_num, fast_v, fast_a, slow_v, slow_a):
        print('////')

        self.rbt_r.io(0, 0)
        if layer_num == 1 or layer_num == 2:
            # 从等待点到点1，接近
            print('//////')
            print('开始')
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean1']['path1'][0], vel=slow_v, acc=slow_a)
            # 到抓取低点
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path1'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 到抓取点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose1'], slow_v, slow_a)

        elif layer_num == 3:
            sys.exit(0)
        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)
            height_offset = 0.0


    def vacuum_clean_by_layer_U625_2(self, layer_num, fast_v, fast_a, slow_v, slow_a):

        if layer_num == 1:
            # moveL回到抓取低点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose0'], slow_v, slow_a)
            # 执行吸盘操作，从抓取低点到纸板上方低点
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean1']['path2'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path2'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 1)
            time.sleep(0.5)
            # moveL到吸取点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose5'], slow_v, slow_a)
            # 力控
            self.Force_Control.move(2)
            time.sleep(2)
            # 从点2到点1，离开
            # moveL到箱子低点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose4'], slow_v, slow_a)
            # 丢弃
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean1']['path4'][0], fast_v, fast_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path4'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 0)
            # 回到抓取低点
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean1']['path5'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path5'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 回到抓取点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose1'], slow_v, slow_a)

        elif layer_num == 2:
            # moveL回到抓取低点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose0'], slow_v, slow_a)
            # 执行吸盘操作，从抓取低点到纸板上方低点
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean2']['path2'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean2']['path2'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 1)
            time.sleep(0.5)
            # moveL到吸取点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean2']['pose5'], slow_v, slow_a)
            # 力控
            self.Force_Control.move(2)
            time.sleep(2)
            # 从点2到点1，离开
            # moveL到箱子低点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean2']['pose4'], slow_v, slow_a)
            # 丢弃
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean2']['path4'][0], fast_v, fast_a)
            # 从点2到点1，离开
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean2']['path4'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.io(0, 0)
            # 回到抓取低点
            self.rbt_r.move_jnts(CONFIG_U625_cc_xipan['vacuum_clean1']['path5'][0], slow_v, slow_a)
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path5'],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            # 回到抓取点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose1'], slow_v, slow_a)

        elif layer_num == 3:
            height_offset = -0.04  # 第三层下降4cm
            print(f"执行第{layer_num}层吸盘清理")
            time.sleep(1)
        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)
            height_offset = 0.0

    def vacuum_clean_by_layer_U625_3(self, layer_num, fast_v, fast_a, slow_v, slow_a):
        print('开始')
        if layer_num == 1 or layer_num == 2:
            # 回到抓取低点
            self.rbt_r.moveL(CONFIG_U625_cc_xipan['vacuum_clean1']['pose0'], slow_v, slow_a)
            # 回到等待点
            self.rbt_r.move_jntspace_path(
                path=CONFIG_U625_cc_xipan['vacuum_clean1']['path1'][::-1],
                interval_time=1.0,
                control_frequency=.002,
                vel=fast_v, acc=fast_a,
                speed_gain=300,
                blend=0.0,
                toppra_vels=[fast_v] * 6,
                toppra_accs=[fast_a] * 6
            )
            self.rbt_r.move_jnts(CONFIG_U1['grasp']['wait_conf'], vel=slow_v, acc=slow_a)

        elif layer_num == 3:
            height_offset = -0.04  # 第三层下降4cm
            print(f"执行第{layer_num}层吸盘清理")
            time.sleep(10)

        else:
            print(f"警告: 未知层数 {layer_num}，使用默认值")
            time.sleep(10)
            height_offset = 0.0



if __name__ == "__main__":
    rbt_r = ur7con.UR5Ag95X_RTDE(
        robot_ip='192.168.125.30',  # 机器人IP
        gp_port='COM5'  # 夹爪端口
    )
    gripper_r = gripper.Ag145driver()
    ForceControl = fcm.ForceControl(rbt_r)
    VC = Vacuum_Clean(rbt_r, ForceControl)
    fast_v = 0.2
    slow_v = 0.1
    fast_a = 0.2
    slow_a = 0.1
    rbt_r.io(0, 0)  # 吸盘停止吸气

    # 手爪打开
    gripper_r.set_force(100)
    gripper_r.jaw_to(100)
    # # 移到抓取点
    # rbt_r.move_jnts(CONFIG_U1['grasp']['wait_conf'],0.1,0.1)
    # layer_num = 1
    # VC.vacuum_clean_by_layer_U1_1(layer_num, fast_v, fast_a, slow_v, slow_a)
    # # 抓取
    # gripper_r.jaw_to(0)
    # # 完成吸盘操作，回到放置位置
    # VC.vacuum_clean_by_layer_U1_2(layer_num, fast_v, fast_a, slow_v, slow_a)
    # # 放手
    # gripper_r.jaw_to(100)
    # # 回到等待点
    # VC.vacuum_clean_by_layer_U1_3(layer_num, fast_v, fast_a, slow_v, slow_a)
    # print("结束程序")

    # 移到抓取点
    rbt_r.move_jnts(CONFIG_U1['grasp']['wait_conf'],0.1,0.1)
    layer_num = 2
    VC.vacuum_clean_by_layer_U625_1(layer_num, fast_v, fast_a, slow_v, slow_a)

    # 抓取
    gripper_r.jaw_to(0)
    # 完成吸盘操作，回到放置位置
    VC.vacuum_clean_by_layer_U625_2(layer_num, fast_v, fast_a, slow_v, slow_a)
    # 放手
    gripper_r.jaw_to(100)
    # 回到等待点
    VC.vacuum_clean_by_layer_U625_3(layer_num, fast_v, fast_a, slow_v, slow_a)
    print("结束程序")
