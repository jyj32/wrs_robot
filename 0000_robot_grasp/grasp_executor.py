import time
import os
import numpy as np
import pickle
from typing import Literal

import robot_con.gofa_con.gofa_con as gofa_con
import drivers.devices.dh.maingripper as dh_r
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76
import robot_sim.robots.gofa5.gofa5 as gf5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

class GraspPlanner:

    def __init__(self, base):
        self.base = base
        self.rbt_s = gf5.GOFA5()
        self.rbt_s.gen_meshmodel().attach_to(self.base)
        self.gripper_s = dh76.Dh76(fingertip_type='r_76')
        gm.gen_frame().attach_to(self.base)
        self.rrtc_s = rrtc.RRTConnect(self.rbt_s)

    def generate_approach_with_shake(self, app_pos, grasp_pos, grasp_rot, start_conf, steps=20, shake_amplitude=0.01, shake_frequency=3):
        """
        生成带有正弦抖动的接近路径关节点
        输入参数：起始位置、抓取位置、抓取旋转矩阵、起始关节配置、步数、抖动幅度和频率。
        返回：一系列关节配置的列表。
        """
        conf_list = []
        for i in range(steps + 1):
            ratio = i / steps
            interp_pos = (1 - ratio) * app_pos + ratio * grasp_pos
            shake_offset = shake_amplitude * np.sin(2 * np.pi * shake_frequency * ratio)
            hand_x_axis = grasp_rot[:, 0]
            interp_pos_with_shake = interp_pos + shake_offset * hand_x_axis
            if i == 0:
                seed = start_conf
            else:
                seed = conf_list[-1]
            conf = self._safe_ik(interp_pos_with_shake, grasp_rot, seed)
            if conf is None:
                print(f"IK failed at step {i}, pos = {interp_pos_with_shake}")
                return None
            conf_list.append(conf)
        return conf_list


    def get_objs_from_box(self, rbt_start_pos, box_pos, box_rot=np.eye(3), grip_z=0.03, approach_dis=0.2,
                          place_pos=np.array([0.798, 0.028, 0.095])):  # 从盒子中获取物体的规划
        """
        输入：
            rbt_start_pos:起始位置
            box_pos:料盒位置
            box_rot:料盒姿态
            grip_z:抓取位置离料盒的位置
            approach_dis:接近距离，两个接近点分别在抓取点和放置点上方20cm处
            place_pos:放置点位置，已改低
        输出：
            conf_dict字典
        """

        grip_pos = np.array([0, 0, grip_z])  # 接近位置
        grip_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)  # 接近角度
        grasp_pos = box_rot @ grip_pos + box_pos    # 抓取位置
        grasp_rot = box_rot @ grip_rot  # 抓取角度，@表示矩阵乘法

        app_pos = grasp_pos - approach_dis * grasp_rot[:, 2]    # 抓取接近点位置
        pick_pos = place_pos + np.array([0, 0, approach_dis])   # 放置位置

        start_conf = self._safe_ik(rbt_start_pos, grasp_rot)    # 起始关节关节点
        app_conf = self._safe_ik(app_pos, grasp_rot, start_conf)    # 抓取接近点位置关节点
        grasp_conf = self._safe_ik(grasp_pos, grasp_rot, app_conf)  # 主抓取位置关节点
        app_to_grip_conf_list = self.generate_approach_with_shake(app_pos + np.array([0, 0, -0.15]), grasp_pos,
                                                                  grasp_rot, app_conf)  # 主接近路径（关节空间路径）
        grasp1_conf = self._safe_ik(grasp_pos, grasp_rot.dot(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)), app_conf)   # 备用抓取位置关节点
        app_to_grip1_conf_list = self.generate_approach_with_shake(app_pos + np.array([0, 0, -0.15]), grasp_pos,
                                                                   grasp_rot.dot(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)),
                                                                   app_conf)    # 备用接近路径关节点，抓取角度不同
        pick_conf = self._safe_ik(pick_pos, grasp_rot, grasp_conf)  # 放置接近点关节点
        place_conf = self._safe_ik(place_pos, grasp_rot, pick_conf)     # 放置位置关节点

        if any(conf is None for conf in (start_conf, app_conf, grasp_conf, pick_conf, place_conf)):     # 任意关节点求取失败
            print("IK 解算失败，流程终止")
            return

        self._visualize_path([start_conf, app_conf, grasp_conf, pick_conf, place_conf], rgba=[0, 1, 0, 0.2])    # 可视化路径

        conf_dict = {
            "start": start_conf,    # 起始关节配置
            "app": app_conf,        # 抓取接近点位置关节点
            "app_to_grip": app_to_grip_conf_list,   # 主接近路径（关节空间路径）
            "grasp": grasp_conf,    # 主抓取位置关节点
            "app_to_grip1": app_to_grip1_conf_list,     # 备用接近路径关节点
            "grasp1": grasp1_conf,  # 备用抓取位置关节点
            "pick": pick_conf,      # 放置接近点关节点
            "place": place_conf     # 放置位置关节点
        }   # 输出字典

        return conf_dict

    def grasp_and_place_single_obj(self, rbt_start_pos, grasp_pos, grasp_rot, place_pos, place_rot, approach_dis=0.1):  # 抓取到放置单个物体
        init_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
        approach_pos = grasp_pos - approach_dis * grasp_rot[:, 2]
        pick_pos = place_pos + np.array([0, 0, approach_dis])
        seed_jnt = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
                             8.44677978e-01, 1.71138133e-01])
        try:
            start_conf = self._safe_ik(rbt_start_pos, init_rot)
            approach_conf = self._safe_ik(approach_pos, grasp_rot, seed_jnt)
            goal_conf = self._safe_ik(grasp_pos, grasp_rot, seed_jnt)
            pick_conf = self._safe_ik(pick_pos, place_rot)
            place_conf = self._safe_ik(place_pos, place_rot)
            confs = [start_conf, approach_conf, goal_conf, pick_conf, place_conf]
            conf_names = ["start", "approach", "goal", "pick", "place"]

            for i, (name, c) in enumerate(zip(conf_names, confs)):
                if c is None:
                    print(f"Cannot solve IK for move '{name}' (index {i})")
                    return

            path_approach = self._safe_plan(start_conf, approach_conf)  # RRT路径规划
            path_grasp = self._get_line_path(approach_pos, grasp_pos, grasp_rot)    # 直线路径规划
            path_app_to_start = self._safe_plan(approach_conf, start_conf)  # RRT路径规划
            path_pick = self._safe_plan(start_conf, pick_conf)  # RRT路径规划
            path_place = self._get_line_path(pick_pos, place_pos, place_rot)    # 直线路径规划
            path_return = self._safe_plan(pick_conf, start_conf)    # RRT路径规划
            path_dict = {"app": path_approach,
                         "grasp": path_grasp,
                         "app_to_start": path_app_to_start,
                         "pick": path_pick,
                         "place": path_place,
                         "return": path_return}

            if not all(path_dict.values()):
                print('Can not plan all path')
                return None

            self._visualize_path(confs, rgba=[0, 0, 1, 0.2])    # 可视化路径
            return path_dict

        except Exception as e:
            print('Exception: ', e)
            return None

    def split_stack_or_stand_objs(self, rbt_start_pos, obj_pos, grip_vec):  # 分开堆叠或站立物体
        center = np.array([0.798, 0.028, 0.13])
        grip_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
        grip_vec = np.array([0, 1, 0]) if grip_vec is None else grip_vec
        if grip_vec[:2].dot((center - obj_pos)[:2]) < 0:
            grip_vec = -grip_vec
        start_pos = obj_pos - 0.03 * grip_vec
        end_pos = obj_pos + 0.03 * grip_vec
        confs = [
            self._safe_ik(rbt_start_pos, grip_rot),
            self._safe_ik(start_pos + np.array([0., 0., 0.1]), grip_rot),
            self._safe_ik(start_pos, grip_rot),
            self._safe_ik(end_pos, grip_rot),
            self._safe_ik(end_pos + np.array([0., 0., 0.1]), grip_rot),
            self._safe_ik(rbt_start_pos, grip_rot)
        ]

        if any(p is None for p in confs):
            print("IK 解算失败")
            return

        self._visualize_path(confs, rgba=[1, 1, 0, 0.2])
        return confs

    def _safe_ik(self, tgt_pos, tgt_rotmat, seed=None, method: Literal['ik', 'tracik'] = 'tracik'):
        #  安全的逆运动学求解。使用ik或tracik求解逆运动学，并检查碰撞。如果求解失败或碰撞，返回None
        #  输入：目标位置，目标3*3旋转矩阵，种子关节角度（作为IK求解的初始值），求解方法'ik'或'tracik'
        #  输出：求解成功且无碰撞，返回关节角度数组（6维，对应6轴机械臂）；求解失败或碰撞，返回None
        seed = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
                         8.44677978e-01, 1.71138133e-01]) if seed is None else seed.copy()  # 如果没有提供种子配置，使用预设的默认关节角度
        if method == 'ik':  # ik求解
            conf = self.rbt_s.ik("arm", tgt_pos, tgt_rotmat, seed_jnt_values=seed)  # 关节角度
        elif method == 'tracik':  # tracik求解
            conf = self.rbt_s.tracik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed)  # 关节角度
        else:  # 方法不对
            raise ValueError("Wrong method")
        if conf is not None:    # 有关节角度
            self.rbt_s.fk("arm", conf)  # 正运动学
            if self.rbt_s.is_collided():    # 机器人碰撞
                return None
        return conf

    def _safe_plan(self, start, goal):  # 使用RRT连接规划器规划路径
        return self.rrtc_s.plan('arm', start, goal, ext_dist=0.03, max_iter=300, max_time=15.0, smoothing_iterations=50)

    def _get_line_path(self, start, goal, rot):  # 生成直线路径，在两个点之间线性插值，并求解每个点的逆运动学。
        return [self._safe_ik(pos, rot) for pos in np.linspace(start, goal, 10)]    # 中间10个点

    def _visualize_path(self, confs, rgba):  # 可视化路径，将每个关节配置的机器人模型以指定颜色添加到基础世界中
        for conf in confs:
            self.rbt_s.fk("arm", conf)
            self.rbt_s.gen_meshmodel(rgba=rgba).attach_to(self.base)


class GraspExecutor:  # 抓取执行
    def __init__(self, move_rbt=True, grip_r=None, rbt_r=None):
        self.move_rbt = move_rbt
        if self.move_rbt:   # 真实机器人运动
            self.gripper_r = grip_r if grip_r else dh_r.MainGripper(port="com4", force=100, speed=30)
            self.rbt_r = rbt_r if rbt_r else gofa_con.GoFaArmController(toggle_debug=False)

    def get_objs_from_box_executor(self, conf_dict):    # 执行从盒子中获取物体的动作序列。按照规划好的关节配置序列控制机器人运动，并控制夹爪开合。
        if self.move_rbt:   # 真实机器人运动
            self.gripper_r.mg_set_force(40)  # 夹爪抓取力为40
            start_time = time.time()    # 记录开始时间
            self.rbt_r.move_j(np.zeros(6))  # 机器人移动到零位
            self.gripper_r.m_gripper.SetTargetSpeed(100)    # 设置夹爪速度为100
            self.gripper_r.jaw_to(0.06)    # 完全张开夹爪76mm，部分张开60mm
            self.rbt_r.move_j(conf_dict["app"], speed_n=300)    # 快速到达接近位置关节点
            self.rbt_r.move_jntspace_path(conf_dict["app_to_grip"], 100)    # 插值运动慢速从接近点到抓取点路径
            self.rbt_r.move_j(conf_dict["grasp"], speed_n=100)  # 精确到达抓取点关节
            self.gripper_r.m_gripper.SetTargetSpeed(40)     # 设置夹爪速度为40
            self.gripper_r.jaw_to(0)    # 闭合夹爪
            self.rbt_r.move_j(conf_dict["app"], speed_n=300)    # 抬起物体
            print(self.gripper_r.m_gripper.GetCurrentPosition())    # 输出夹爪位置
            if self.gripper_r.m_gripper.GetCurrentPosition() == 0:  # 夹爪位置为0，抓取失败，尝试备用方案
                self.gripper_r.jaw_to(0.06)    # 张开夹爪76mm
                self.rbt_r.move_jntspace_path(conf_dict["app_to_grip1"], 100)   #
                self.rbt_r.move_j(conf_dict["grasp1"], speed_n=100)  #
                self.gripper_r.jaw_to(0)    # 闭合夹爪
            self.rbt_r.move_j(conf_dict["pick"], speed_n=300)   # 抓起路径
            self.rbt_r.move_j(conf_dict["place"], speed_n=300)  # 放置路径
            self.gripper_r.m_gripper.SetTargetSpeed(50) # 设置夹爪速度为50
            self.gripper_r.jaw_to(0.06)    # 张开夹爪76mm
            self.rbt_r.move_j(conf_dict["pick"], speed_n=300)   # 抓起路径
            self.rbt_r.move_j(conf_dict["start"], speed_n=300)
            print("执行耗时：", time.time() - start_time)

    def grasp_single_obj_executor(self, path_dict):     # 抓单个物体
        high_rbt_speed = 200
        low_rbt_speed = 100
        high_grip_speed = 100
        low_grip_speed = 40
        if self.move_rbt:
            start_time = time.time()
            self.gripper_r.mg_set_force(30)
            self.gripper_r.m_gripper.SetTargetSpeed(high_grip_speed)
            self.rbt_r.move_jntspace_path(path_dict["app"], high_rbt_speed)
            self.gripper_r.jaw_to(0.015)    # 张开至3cm？
            self.rbt_r.move_jntspace_path(path_dict["grasp"], low_rbt_speed)
            self.gripper_r.m_gripper.SetTargetSpeed(low_grip_speed)
            self.gripper_r.jaw_to(0)
            self.rbt_r.move_jntspace_path(path_dict["grasp"][::-1], high_rbt_speed)
            self.rbt_r.move_jntspace_path(path_dict["app_to_start"], high_rbt_speed)
            print("执行耗时：", time.time() - start_time)
            return True
        return False

    def place_single_obj_executor(self, path_dict):     # 放单个物体
        high_rbt_speed = 200
        low_rbt_speed = 100
        high_grip_speed = 100
        low_grip_speed = 40
        if self.move_rbt:
            start_time = time.time()
            self.rbt_r.move_jntspace_path(path_dict["pick"], high_rbt_speed)
            self.rbt_r.move_jntspace_path(path_dict["place"], low_rbt_speed)
            self.gripper_r.m_gripper.SetTargetSpeed(low_grip_speed)
            self.gripper_r.jaw_to(0.02)
            self.rbt_r.move_jntspace_path(path_dict["place"][::-1], high_rbt_speed)
            self.rbt_r.move_jntspace_path(path_dict["return"], high_rbt_speed)
            print("执行耗时：", time.time() - start_time)
            print("Grasp and place successfully completed")
            return True
        return False

    def split_stack_or_stand_objs_executor(self, conf_list):    # 分开物体
        if self.move_rbt:
            start_time = time.time()
            self.gripper_r.m_gripper.SetTargetSpeed(100)
            for i, conf in enumerate(conf_list):
                if i == 2:
                    self.gripper_r.jaw_to(0)
                    self.rbt_r.move_j(conf, speed_n=200)
                elif i == 3:
                    self.rbt_r.move_j(conf, speed_n=300)
                else:
                    self.rbt_r.move_j(conf, speed_n=300)
            print("执行耗时：", time.time() - start_time)
            return True
        return False


if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    planner = GraspPlanner(base)    # 抓取规划
    executor = GraspExecutor(move_rbt=False)    # 抓取执行
    base.run()
