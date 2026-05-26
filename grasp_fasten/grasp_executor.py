import copy
import math
import time
import os
import numpy as np
import pickle
from typing import Literal
import robot_con.gofa_con.gofa_con as gofa_con
import robot_con.ur.ur7_dh50_rtde as ur7e_con
import drivers.devices.dh.maingripper as dh_r
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76
import robot_sim.robots.gofa5.gofa5 as gf5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_sim.robots.robot_interface as ri
import robot_sim.end_effectors.gripper.gripper_interface as gp
from typing import Dict, List, Optional, Tuple
from config import CONFIG

class GraspPlanner: # 路径规划
    # 初始化
    def __init__(self, base, robot_sim: ri.RobotInterface, gripper_sim: gp.GripperInterface):
        self.base = base
        self.rbt_s = robot_sim
        self.rbt_s.gen_meshmodel().attach_to(self.base)
        self.gripper_s = gripper_sim
        gm.gen_frame().attach_to(self.base)
        self.rrtc_s = rrtc.RRTConnect(self.rbt_s)

    # 振动接近路径
    def generate_approach_with_shake(self, app_pos, grasp_pos, grasp_rot, start_conf, steps=20, shake_amplitude=0.01,
                                     shake_frequency=3):
        conf_list = []
        for i in range(steps + 1):
            ratio = i / steps
            interp_pos = (1 - ratio) * app_pos + ratio * grasp_pos
            shake_offset = shake_amplitude * np.sin(2 * np.pi * shake_frequency * ratio)
            hand_x_axis = grasp_rot[:, 1]
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

    # 锯齿形路径点规划，用于分开堆叠物体
    def plan_z_path(self, start_pos, end_pos, rot, offset, num):
        '''
            输入:
                start_pos:起始位置
                end_pos:终点位置
                rot:旋转矩阵
                offset:偏移距离
                num:中间点的数量
            输出:
                confs:锯齿形路径点
        '''
        vec = end_pos - start_pos
        end_conf = self._safe_ik(end_pos, rot)  # 终点关节点
        if end_conf is None:
            print(f"终点ik解算失败，终止")
            return None
        # 起点到终点的直线距离
        length = np.linalg.norm(vec)
        if length < 1e-6:   # 直线距离过短
            return [end_conf]
        # 中间点间隔
        xy_dieta = vec/(num+1)
        vx, vy = vec[0], vec[1]
        # 与路径方向垂直的单位向量(xy平面内)
        perp_dir = np.array([-vy, vx, 0.0]) / np.hypot(vx, vy)
        present_point = start_pos
        present_conf = self._safe_ik(present_point, rot)
        if present_conf is None:
            print(f"起点ik解算失败，终止")
            return None
        confs = []
        for i in range(num):
            if i%2 == 1:    # i为奇数
                point = present_point + xy_dieta + perp_dir*offset
            else:
                point = present_point + xy_dieta - perp_dir*offset
            conf = self._safe_ik(point, rot, seed = present_conf)
            if conf is None:
                print(f"第{i}个中间点IK解算失败，终止")
                return None
            present_point = point
            present_conf = conf
            confs.append(conf)
        confs.append(end_conf)  # 加上终点
        return confs

    # 从箱中抓取物体路径
    def get_objs_from_box(self, rbt_start_pos, box_pos, box_rot=np.eye(3), grip_z=0.03, approach_dis=0.2,
                          place_pos=np.array([0.798, 0.028, 0.13])):
        # grip_pos = np.array([0, 0, grip_z])
        grip_rot = CONFIG['robot']['default_rot']
        # grasp_pos = box_rot @ grip_pos + box_pos
        # grasp_rot = box_rot @ grip_rot
        grasp_pos = box_pos
        grasp_rot = box_rot @ grip_rot

        app_pos = grasp_pos - approach_dis * grasp_rot[:, 2]
        pick_pos = place_pos + np.array([0, 0, approach_dis])

        start_conf = self._safe_ik(rbt_start_pos, grasp_rot)
        app_conf = self._safe_ik(app_pos, grasp_rot, start_conf)
        grasp_conf = self._safe_ik(grasp_pos, grasp_rot, app_conf)
        app_to_grip_conf_list = self.generate_approach_with_shake(grasp_pos + np.array([0, 0, 0.05]), grasp_pos,
                                                                  grasp_rot, app_conf)
        app1_conf = self._safe_ik(app_pos, grasp_rot.dot(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)), start_conf)
        grasp1_conf = self._safe_ik(grasp_pos, grasp_rot.dot(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)), app1_conf)
        app1_to_grip1_conf_list = self.generate_approach_with_shake(app_pos + np.array([0, 0, -0.15]), grasp_pos,
                                                                   grasp_rot.dot(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)),
                                                                   app_conf)
        pick_conf = self._safe_ik(pick_pos, grasp_rot, grasp_conf)
        place_conf = self._safe_ik(place_pos, grasp_rot, pick_conf)

        if any(conf is None for conf in (start_conf, app_conf, grasp_conf, pick_conf, place_conf)):
            print("IK 解算失败，流程终止")
            return

        self._visualize_path([start_conf, app_conf, grasp_conf, pick_conf, place_conf], rgba=[0, 1, 0, 0.2])

        conf_dict = {
            "start": start_conf,
            "app": app_conf,
            "shake_start": app_to_grip_conf_list[0],
            "app_to_grip": app_to_grip_conf_list,
            "grasp": grasp_conf,
            "app1": app1_conf,
            "app1_to_grip1": app1_to_grip1_conf_list,
            "grasp1": grasp1_conf,
            "pick": pick_conf,
            "place": place_conf
        }

        return conf_dict

    # 抓取单个物体路径
    def grasp_and_place_single_obj(self, rbt_start_pos, grasp_pos, grasp_rot, place_pos, place_rot, approach_dis=0.1):
        # init_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
        init_rot = CONFIG['robot']['default_rot']
        approach_pos = grasp_pos - approach_dis * grasp_rot[:, 2]
        pick_pos = place_pos + np.array([0, 0, approach_dis])
        # seed_jnt = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
        #                      8.44677978e-01, 1.71138133e-01])
        seed_jnt = None
        try:
            start_conf = self._safe_ik(rbt_start_pos, init_rot) # 开始位姿
            approach_conf = self._safe_ik(approach_pos, grasp_rot, seed_jnt)    # 接近位姿
            goal_conf = self._safe_ik(grasp_pos, grasp_rot, seed_jnt)
            print(pick_pos)
            pick_conf = self._safe_ik(pick_pos, place_rot)
            place_conf = self._safe_ik(place_pos, place_rot)
            confs = [start_conf, approach_conf, goal_conf, pick_conf, place_conf]
            conf_names = ["start", "approach", "goal", "pick", "place"]

            for i, (name, c) in enumerate(zip(conf_names, confs)):
                if c is None:
                    print(f"Cannot solve IK for move '{name}' (index {i})")
                    return

            path_approach = self._safe_plan(start_conf, approach_conf)
            path_grasp = self._get_line_path(approach_pos, grasp_pos, grasp_rot)
            path_app_to_start = self._safe_plan(approach_conf, start_conf)
            path_pick = self._safe_plan(start_conf, pick_conf)
            path_place = self._get_line_path(pick_pos, place_pos, place_rot)
            path_return = self._safe_plan(pick_conf, start_conf)
            path_dict = {"app": path_approach,
                         "grasp": path_grasp,
                         "app_to_start": path_app_to_start,
                         "pick": path_pick,
                         "place": path_place,
                         "return": path_return}
            if not all(path_dict.values()):
                print('Can not plan all path')
                return None

            self._visualize_path(confs, rgba=[0, 0, 1, 0.2])
            return path_dict

        except Exception as e:
            print('Exception: ', e)
            return None

    # 分开直立物体
    def split_stand_objs(self, rbt_start_pos, obj_pos, grip_vec):
        '''
        输入：
            rbt_start_pos:机械臂初始位置
            obj_pos:物体位置
            grip_vec:推动方向
        输出：
            confs:推物体路径点
        '''
        center = CONFIG['plate']['center']  # 海绵垫中心
        # grip_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
        grip_rot = CONFIG['robot']['default_rot']   # 机器人姿态
        grip_vec = np.array([0, 1, 0]) if grip_vec is None else grip_vec    # 推动方向
        # 调整推动方向（确保推向中心）
        if grip_vec[:2].dot((center - obj_pos)[:2]) < 0:    # 点积为负，说明为两向量钝角
            grip_vec = -grip_vec    # 改成推向中心
        start_pos = obj_pos - 0.03 * grip_vec   # 开始位置
        end_pos = obj_pos + 0.03 * grip_vec # 结束位置
        # 调整机器人姿态，手爪应与推动方向垂直
        theta = np.arctan2(abs(grip_vec[1]), abs(grip_vec[0]))  # 平面向量角,范围 [0, π/2]
        print(f"旋转角：{theta}")
        grasp_rot_z = rm.rotmat_from_axangle([0, 0, 1], theta)
        grip_rot2 = np.dot(grasp_rot_z, grip_rot)

        rbt_start_conf = self._safe_ik(rbt_start_pos, grip_rot)  # 机器人起始位姿
        start_up_conf = self._safe_ik(start_pos + np.array([0., 0., 0.1]), grip_rot2)  # 开始位姿上方10cm
        start_conf = self._safe_ik(start_pos, grip_rot2)  # 开始位姿
        end_conf = self._safe_ik(end_pos, grip_rot2)  # 结束位姿
        end_up_conf = self._safe_ik(end_pos + np.array([0., 0., 0.1]), grip_rot2)  # 结束位姿上方

        confs = [
            rbt_start_conf, # 机器人起始位姿
            start_up_conf,   # 开始位姿上方10cm
            start_conf, # 开始位姿
            end_conf,   # 结束位姿
            end_up_conf, # 结束位姿上方
            rbt_start_conf  # 起始位姿
        ]

        if any(p is None for p in confs):
            print("IK 解算失败")
            return

        self._visualize_path(confs, rgba=[1, 1, 0, 0.2])
        return confs

    # 只分开堆叠物体
    def split_stack_objs(self, rbt_start_pos, obj_pos, grip_vec):
        '''
        只分开堆叠物体
        输入：
            rbt_start_pos:机械臂初始位置
            obj_pos:物体位置
            grip_vec:推动方向，即矩形框的短边方向，指向海绵平台中心
        输出：
            confs:推物体路径点
        '''
        center = CONFIG['plate']['center']  # 海绵垫中心
        grip_rot = CONFIG['robot']['default_rot']   # 机器人默认姿态，也是拍照姿态
        grip_vec = np.array([0, 1, 0]) if grip_vec is None else grip_vec    # 推动方向
        # 调整推动方向（确保推向中心）
        if grip_vec[:2].dot((center - obj_pos)[:2]) < 0:    # 点积为负，说明为两向量钝角
            grip_vec = -grip_vec    # 改成推向中心
        start_pos = obj_pos - 0.03 * grip_vec   # 开始位置
        end_pos = obj_pos + 0.03 * grip_vec # 结束位置
        # 调整机器人姿态，手爪应与推动方向垂直
        theta = np.arctan2(abs(grip_vec[1]), abs(grip_vec[0]))  # 方向向量直线与x轴直线的平面线线角,范围 [0, π/2]
        if grip_vec[1]*grip_vec[0]<0:   # 2，4象限时为负号角
            theta = -theta
        print(f"绕z轴旋转角：{theta}")
        grip_rot2 = np.dot(rm.rotmat_from_axangle([0, 0, 1], theta), grip_rot)

        rbt_start_conf = self._safe_ik(rbt_start_pos, grip_rot)  # 机器人起始位姿
        if rbt_start_conf is None:
            print("rbt_start_conf IK 解算失败")
            return None
        start_up_conf = self._safe_ik(start_pos + np.array([0., 0., 0.1]), grip_rot2)  # 开始位姿上方10cm
        if start_up_conf is None:
            print("start_up_conf IK 解算失败")
            return None
        start_conf = self._safe_ik(start_pos, grip_rot2)  # 开始位姿
        if start_conf is None:
            print("start_conf IK 解算失败")
            return None
        obj_conf = self._safe_ik(obj_pos, grip_rot2)  # 到达物体的位姿
        if obj_conf is None:
            print("obj_conf IK 解算失败")
            return None
        z_path = self.plan_z_path(obj_pos, end_pos, grip_rot2, 0.015, 4)    # z形路径点
        if z_path is None:
            print("z_path IK 解算失败")
            return None
        end_up_conf = self._safe_ik(end_pos + np.array([0., 0., 0.1]), grip_rot2)  # 结束位姿上方
        if end_up_conf is None:
            print("end_up_conf IK 解算失败")
            return None
        confs = {
            "start": rbt_start_conf,  # 起始位姿（高速）
            "approach_up": start_up_conf,  # 上升至开始位姿上方（高速）
            "approach": start_conf,  # 下降至开始位姿（高速）
            "to_obj": obj_conf,  # 移动至物体位置（低速）
            "z_path": z_path,  # 锯齿形路径点列表（低速依次执行）
            "up": end_up_conf,  # 结束位姿上升（高速）
            "back": rbt_start_conf  # 返回初始位姿（高速）
        }
        return confs

    def _safe_ik(self, tgt_pos, tgt_rotmat, seed=None, method: Literal['ik', 'tracik'] = 'ik'):
        # seed = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
        #                  8.44677978e-01, 1.71138133e-01]) if seed is None else seed.copy()
        if method == 'ik':
            conf = self.rbt_s.ik("arm", tgt_pos, tgt_rotmat, seed_jnt_values=seed)
        elif method == 'tracik':
            conf = self.rbt_s.tracik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed)
        else:
            raise ValueError("Wrong method")
        if conf is not None:
            self.rbt_s.fk("arm", conf)
            if self.rbt_s.is_collided():
                return None
        if conf is None:
            print(f"ik求解失败：位置{tgt_pos}姿态{tgt_rotmat}")

        return conf

    def _safe_plan(self, start, goal):
        return self.rrtc_s.plan('arm', start, goal, ext_dist=0.06, max_iter=300, max_time=15.0, smoothing_iterations=50)

    def _get_line_path(self, start, goal, rot): # 返回关节点路径
        return [self._safe_ik(pos, rot) for pos in np.linspace(start, goal, 10)]

    def _visualize_path(self, confs, rgba):
        for conf in confs:
            self.rbt_s.fk("arm", conf)
            self.rbt_s.gen_meshmodel(rgba=rgba).attach_to(self.base)


class GraspExecutor:    # 抓取执行abb机器人

    def __init__(self, move_rbt=True, grip_r=None, rbt_r=None):
        self.move_rbt = move_rbt
        if self.move_rbt:
            self.gripper_r = grip_r if grip_r else dh_r.MainGripper(port="com4", force=100, speed=30)
            self.rbt_r = rbt_r if rbt_r else gofa_con.GoFaArmController(toggle_debug=False)

    def get_objs_from_box_executor(self, conf_dict):
        if self.move_rbt:
            self.gripper_r.mg_set_force(100)
            start_time = time.time()
            self.rbt_r.move_j(np.zeros(6))
            self.gripper_r.m_gripper.SetTargetSpeed(100)
            self.gripper_r.jaw_to(0.076)
            self.rbt_r.move_j(conf_dict["app"], speed_n=300)
            self.rbt_r.move_jntspace_path(conf_dict["app_to_grip"], 100)
            self.rbt_r.move_j(conf_dict["grasp"], speed_n=100)
            self.gripper_r.m_gripper.SetTargetSpeed(40)
            self.gripper_r.jaw_to(0)
            self.rbt_r.move_j(conf_dict["app"], speed_n=300)
            print(self.gripper_r.m_gripper.GetCurrentPosition())
            if self.gripper_r.m_gripper.GetCurrentPosition() == 0:
                self.gripper_r.jaw_to(0.076)
                self.rbt_r.move_jntspace_path(conf_dict["app1_to_grip1"], 100)
                self.rbt_r.move_j(conf_dict["grasp1"], speed_n=100)
                self.gripper_r.jaw_to(0)
                self.rbt_r.move_j(conf_dict["app1"], speed_n=300)
            self.rbt_r.move_j(conf_dict["pick"], speed_n=300)
            self.rbt_r.move_j(conf_dict["place"], speed_n=300)
            self.gripper_r.m_gripper.SetTargetSpeed(50)
            self.gripper_r.jaw_to(0.076)
            self.rbt_r.move_j(conf_dict["pick"], speed_n=300)
            self.rbt_r.move_j(conf_dict["start"], speed_n=300)
            print("执行耗时：", time.time() - start_time)

    def grasp_single_obj_executor(self, path_dict):
        high_rbt_speed = 150
        low_rbt_speed = 100
        high_grip_speed = 100
        low_grip_speed = 40
        if self.move_rbt:
            start_time = time.time()
            self.gripper_r.mg_set_force(30)
            self.gripper_r.m_gripper.SetTargetSpeed(high_grip_speed)
            self.rbt_r.move_jntspace_path(path_dict["app"], high_rbt_speed)
            self.gripper_r.jaw_to(0.02)
            self.rbt_r.move_jntspace_path(path_dict["grasp"], low_rbt_speed)
            self.gripper_r.m_gripper.SetTargetSpeed(low_grip_speed)
            self.gripper_r.jaw_to(0)
            self.rbt_r.move_jntspace_path(path_dict["grasp"][::-1], high_rbt_speed)
            self.rbt_r.move_jntspace_path(path_dict["app_to_start"], high_rbt_speed)
            print("执行耗时：", time.time() - start_time)
            return True
        return False

    def place_single_obj_executor(self, path_dict):
        high_rbt_speed = 150
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

    def split_stack_or_stand_objs_executor(self, conf_list):
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


class GraspExecutorRTDE:    # 抓取执行，ur7e机器人

    def __init__(self,
                 robot_real: ur7e_con.UR5Ag95X_RTDE,
                 move_rbt:  bool = True):
        self.move_rbt = move_rbt
        self.rbt_r = robot_real

    def get_objs_from_box_executor(self, conf_dict):    # 从盒中抓取物体
        gripper_speed = 100
        gripper_force = 10
        rbt_speed = 2.5
        rbt_acc = 2.5
        if self.move_rbt:
            start_time = time.time()
            # self.rbt_r.move_jnts(np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi]), vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.open_gripper_dh(gripper_speed, gripper_force)
            # self.rbt_r.move_jnts(conf_dict["app"], vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.move_jnts(conf_dict["shake_start"], vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.move_jntspace_path(conf_dict["app_to_grip"], vel=0.5)
            # self.rbt_r.move_jnts(conf_dict["grasp"], vel=rbt_speed/2, acc=rbt_acc)
            # self.rbt_r.close_gripper_dh(gripper_speed/2, gripper_force)
            # self.rbt_r.move_jnts(conf_dict["app"], vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.move_jnts(conf_dict["pick"], vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.move_jnts(conf_dict["place"], vel=rbt_speed, acc=rbt_acc)
            # self.rbt_r.open_gripper_dh(gripper_speed/2, gripper_force)
            # self.rbt_r.move_jnts(conf_dict["pick"], vel=rbt_speed, acc=rbt_acc)
            self.rbt_r.move_jnts(conf_dict["start"], vel=rbt_speed, acc=rbt_acc)    # 回到拍摄点
            print("执行耗时：", time.time() - start_time)

    def grasp_single_obj_executor(self, path_dict):
        # 执行抓取
        high_rbt_speed = 2.5
        low_rbt_speed = 1.0
        rbt_acc = 2.5
        high_toppra_vels = [high_rbt_speed]*6
        low_rbt_speed = [low_rbt_speed]*6
        toppra_accs = [rbt_acc]*6

        high_grip_speed = 100
        low_grip_speed = 40
        if self.move_rbt:
            start_time = time.time()
            print("手爪张开")
            # self.rbt_r.close_to_dh(0.03)
            self.rbt_r.close_to_dh(25)  # 宽度范围是0-100
            path1 = copy.deepcopy(path_dict['app']) # 深拷贝，后续的操作不会影响原数据
            path1.extend(path_dict['grasp'])
            self.rbt_r.move_jntspace_path(path1, toppra_vels=high_toppra_vels, toppra_accs=toppra_accs)
            self.rbt_r.close_gripper_dh(speed=low_grip_speed, force=20)
            print("手爪闭合")
            path2 = copy.deepcopy(path_dict['grasp'][::-1])
            path2.extend(path_dict['app_to_start'])
            self.rbt_r.move_jntspace_path(path2, toppra_vels=high_toppra_vels, toppra_accs=toppra_accs)
            # self.rbt_r.move_jntspace_path(path_dict["app_to_start"], toppra_vels=high_toppra_vels)
            print("抓取执行耗时：", time.time() - start_time)
            return True
        return False

    def place_single_obj_executor(self, path_dict):
        # 放置单个物体

        high_rbt_speed = 2.5
        low_rbt_speed = 1.0
        rbt_acc = 2.5
        high_toppra_vels = [high_rbt_speed] * 6
        low_toppra_speed = [low_rbt_speed] * 6
        toppra_accs = [rbt_acc] * 6
        high_grip_speed = 100
        low_grip_speed = 40
        if self.move_rbt:
            start_time = time.time()
            path1 = copy.deepcopy(path_dict['pick'])
            # path1.extend(path_dict['place'])
            self.rbt_r.move_jntspace_path(path1, toppra_vels=high_toppra_vels, toppra_accs=toppra_accs)
            self.rbt_r.move_jntspace_path(path_dict["place"], toppra_vels=low_toppra_speed, toppra_accs=toppra_accs)
            # time.sleep(10)
            self.rbt_r.close_to_dh(20)  # 张开手爪
            path2 = copy.deepcopy(path_dict['place'][::-1])
            path2.extend(path_dict['return'])
            self.rbt_r.move_jntspace_path(path2, toppra_vels=high_toppra_vels, toppra_accs=toppra_accs)
            print("执行耗时：", time.time() - start_time)
            print("Grasp and place successfully completed")
            return True
        return False

    # 分开直立物体
    def split_stand_objs_executor(self, conf_list):
        high_rbt_speed = 2.5
        low_rbt_speed = 1.0
        rbt_acc = 2.5
        if self.move_rbt:
            start_time = time.time()
            for i, conf in enumerate(conf_list):
                # 机器人起始位姿，开始位姿上方10cm，开始位姿，结束位姿，结束位姿上方，起始位姿
                if i == 2:
                    self.rbt_r.close_gripper_dh()   # 闭合手爪
                    self.rbt_r.move_jnts(conf, vel=high_rbt_speed, acc=rbt_acc) # 移到开始位姿
                elif i == 3:
                    self.rbt_r.move_jnts(conf, vel=low_rbt_speed, acc=rbt_acc)  # 移到结束位姿，慢速
                else:   # i为0，1，4，5
                    self.rbt_r.move_jnts(conf, vel=high_rbt_speed, acc=rbt_acc)
            print("分开直立物体动作执行耗时：", time.time() - start_time)
            return True
        return False

    # 只分开堆叠物体
    def split_stack_objs_executor(self, conf):
        high_rbt_speed = 2.5
        low_rbt_speed = 1.0
        rbt_acc = 2.5
        if self.move_rbt:
            start_time = time.time()
            self.rbt_r.move_jnts(conf["start"], vel=high_rbt_speed, acc=rbt_acc)  # 移到开始位姿
            self.rbt_r.move_jnts(conf["approach_up"], vel=high_rbt_speed, acc=rbt_acc)  # 接近位姿
            self.rbt_r.close_gripper_dh()  # 闭合手爪
            self.rbt_r.move_jnts(conf["approach"], vel=high_rbt_speed, acc=rbt_acc)  # 起点位姿
            self.rbt_r.move_jnts(conf["to_obj"], vel=high_rbt_speed, acc=rbt_acc)  # 移到物体位置
            for jnts in conf["z_path"]:
                self.rbt_r.move_jnts(jnts, vel=low_rbt_speed, acc=rbt_acc)  # z形路径
            self.rbt_r.move_jnts(conf["up"], vel=high_rbt_speed, acc=rbt_acc)  # 上方位姿
            self.rbt_r.move_jnts(conf["start"], vel=high_rbt_speed, acc=rbt_acc)  # 开始位姿
            print("分开堆叠物体动作执行耗时：", time.time() - start_time)
            return True
        return False

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    planner = GraspPlanner(base, gf5.GOFA5(), dh76.Dh76())
    executor = GraspExecutor(move_rbt=False)
    base.run()
