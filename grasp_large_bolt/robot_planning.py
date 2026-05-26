import copy
import time
import os
import numpy as np
import pickle
from typing import Literal

from ultralytics.utils.callbacks import base

import robot_sim.robots.ur7e.ur7e_withouttable as rbt
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd
import robot_con.ur.ur7_dh50_rtde as ur7e_con
import drivers.devices.dh.maingripper as dh_r
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_sim.robots.robot_interface as ri
import robot_sim.end_effectors.gripper.gripper_interface as gp
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

from grasp_planner import GraspPlanner
from config import CONFIG


class RobotPlanner:
    def __init__(self, robot_sim, robot_real, obstacle_list, gripper_sim, config, base=None, sim=True):
        self.base = base
        self.robot_sim = rbt.UR7E() if robot_sim is None else robot_sim
        if not sim:
            self.robot_real = robot_real
        self.gripper = hnd.Dh50() if gripper_sim is None else gripper_sim
        self.sim = sim
        self.config = config
        self.rrtc_s = rrtc.RRTConnect(self.robot_sim)
        self.obstacle_list = obstacle_list if obstacle_list is not None else self.robot_sim.get_obstacle_list(base, True)

        self.high_speed = config['robot']['high_speed']
        self.low_speed = config['robot']['low_speed']
        self.acc = config['robot']['acc']
        self.high_toppra = [self.high_speed] * 6
        self.low_toppra = [self.low_speed] * 6
        self.toppra_accs = [self.acc] * 6

        self.grip_speed = config['gripper']['low_speed']
        self.grip_force = config['gripper']['force']

        self.obj_model = cm.CollisionModel(f'./object/{CONFIG["obj_name"]}.stl')
        self.shelf_model = cm.CollisionModel(f'./object/{CONFIG["shelf_name"]}.stl')
        self.target_model = cm.CollisionModel(f'./object/{CONFIG["target_name"]}.stl')

    def init_rbt(self):
        self.robot_sim.fk("arm", self.robot_sim.arm_homeconf)
        self.robot_sim.gen_meshmodel().attach_to(self.base)
        if not self.sim:
            self.robot_real.open_gripper_dh(speed=self.grip_speed, force=self.grip_force)
            self.robot_real.move_jnts(self.robot_sim.arm_homeconf, self.high_speed, self.acc)

    def _safe_ik(self, tgt_pos, tgt_rotmat, seed=None, method: Literal['ik', 'tracik'] = 'ik'):
        if method == 'ik':
            conf = self.robot_sim.ik("arm", tgt_pos, tgt_rotmat, seed_jnt_values=seed)
        elif method == 'tracik':
            conf = self.robot_sim.tracik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed)
        else:
            raise ValueError("Wrong method")
        if conf is not None:
            self.robot_sim.fk("arm", conf)
            if self.robot_sim.is_collided(obstacle_list = self.obstacle_list):
                self.robot_sim.gen_meshmodel().attach_to(self.base)
                self.base.run()
                print("is_collided")
                return None
        return conf

    def _safe_plan(self, start, goal):
        return self.rrtc_s.plan('arm', start, goal, obstacle_list=self.obstacle_list,
                                ext_dist=0.06, max_iter=300, max_time=15.0, smoothing_iterations=50)

    def _get_line_path(self, start, goal, rot):
        return [self._safe_ik(pos, rot) for pos in np.linspace(start, goal, 10)]

    def _visualize_path(self, paths, rgba):
        i = 0
        for path in paths.values():
            for conf in path:
                if i % 50 == 0:
                    self.robot_sim.jaw_to('arm', 0.05)
                    self.robot_sim.fk("arm", conf)
                    self.robot_sim.gen_meshmodel(rgba=rgba).attach_to(self.base)

    def plan_pick_place(self, object_pose, target_pose, grasp_planner: GraspPlanner, visualize=True):
        """
        返回：dict, dict, grasp_info
        pick_path_dict, place_path_dict, grasp_info
        """
        # 拿到抓取姿态
        grasp_infos = grasp_planner.plan_grasp(manual_set=True)
        for grasp_info in grasp_infos:
            pick_path = self._plan_pick(object_pose, grasp_info)
            if pick_path is None:
                continue
            place_path = self._plan_place(pick_path['pick'][-1], target_pose)
            if place_path is None:
                continue
            if visualize:
                self.obj_model.set_homomat(object_pose)
                self.obj_model.set_rgba([0, 1, 1, 1])
                self.obj_model.attach_to(self.base)
                obj_tgt_model = self.obj_model.copy()
                obj_tgt_model.set_homomat(target_pose)
                obj_tgt_model.set_rgba([1, 0, 0, 1])
                obj_tgt_model.attach_to(self.base)
                self._visualize_path(pick_path, [0, 1, 0, 0.5])
                self._visualize_path(place_path, [0, 0, 1, 0.5])
            return pick_path, place_path, grasp_info
        return None, None, None

    def _plan_pick(self, object_pose, grasp_info):
        bolt_pos, bolt_rot = object_pose[:3, 3], object_pose[:3, :3]
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        # 抓取位置
        bolt_grasp_pos = bolt_rot.dot(jaw_center_pos) + bolt_pos
        print(f"bolt_grasp_pos: {bolt_grasp_pos}")
        # 抓取旋转矩阵
        bolt_grasp_rot = bolt_rot.dot(jaw_center_rotmat)
        approach_pos = bolt_grasp_pos - bolt_grasp_rot[:, 2] * self.config["approach_dist"]

        pick_pos = bolt_grasp_pos + self.config["lift_dist"]

        start_conf = self.robot_sim.arm_homeconf
        approach_conf = self._safe_ik(approach_pos, bolt_grasp_rot)
        grasp_conf = self._safe_ik(bolt_grasp_pos, bolt_grasp_rot)
        pick_conf = self._safe_ik(pick_pos, bolt_grasp_rot)

        confs = [start_conf, approach_conf, grasp_conf, pick_conf]
        conf_names = ["start", "app", "grasp", "pick"]
        for i, (name, c) in enumerate(zip(conf_names, confs)):
            if c is None:
                print(f"[FAIL][_plan_pick] IK求解失败: '{name}' (index {i})，参数：")
                if name == "app":
                    print(f"  approach_pos: {approach_pos}\n  approach_rot:\n{bolt_grasp_rot}")
                elif name == "grasp":
                    print(f"  grasp_pos: {bolt_grasp_pos}\n  grasp_rot:\n{bolt_grasp_rot}")
                elif name == "pick":
                    print(f"  pick_pos: {pick_pos}\n  pick_rot:\n{bolt_grasp_rot}")
                return None

        path_app = self._safe_plan(start_conf, approach_conf)
        if not path_app:
            print(f"[FAIL][_plan_pick] 起始到approach路径规划失败")
            return None
        path_grasp = self._get_line_path(approach_pos, bolt_grasp_pos, bolt_grasp_rot)
        if not path_grasp:
            print(f"[FAIL][_plan_pick] approach到grasp插值路径失败")
            return None
        path_pick = self._get_line_path(bolt_grasp_pos, pick_pos, bolt_grasp_rot)
        if not path_pick:
            print(f"[FAIL][_plan_pick] grasp到pick插值路径失败")
            return None
        return {
            "app": path_app,
            "grasp": path_grasp,
            "pick": path_pick
        }

    def _plan_place(self, pick_conf, target_pose):
        start_conf = self.robot_sim.arm_homeconf
        target_pos, target_rot = target_pose[:3, 3], target_pose[:3, :3]
        approach_pos = target_pos - self.config["approach_dist"] * target_rot[:, 2]
        app_conf = self._safe_ik(approach_pos, target_rot)
        # goal_conf = self._safe_ik(target_pos, target_rot)

        confs = [pick_conf, app_conf]
        # confs = [pick_conf, app_conf, goal_conf]
        conf_names = ["pick_conf", "app"]
        # conf_names = ["pick_conf", "app", "goal"]
        for i, (name, c) in enumerate(zip(conf_names, confs)):
            if c is None:
                print(f"[FAIL][_plan_place] IK求解失败: '{name}' (index {i})，参数：")
                if name == "app":
                    print(f"  approach_pos: {approach_pos}\n  approach_rot:\n{target_rot}")
                elif name == "goal":
                    print(f"  goal_pos: {target_pos}\n  goal_rot:\n{target_rot}")
                return None

        path_app = self._safe_plan(pick_conf, app_conf)
        if not path_app:
            print(f"[FAIL][_plan_place] pick到approach路径规划失败")
            return None
        # path_goal = self._get_line_path(approach_pos, target_pos, target_rot)
        # if not path_goal:
        #     print(f"[FAIL][_plan_place] approach到goal插值路径失败")
        #     return None
        # path_place = self._safe_plan(goal_conf, app_conf)
        # if not path_place:
        #     print(f"[FAIL][_plan_place] goal到approach路径规划失败")
        #     return None
        path_return = self._safe_plan(app_conf, start_conf)
        if not path_return:
            print(f"[FAIL][_plan_place] return路径规划失败")
            return None
        return {
            "app": path_app,
            # "goal": path_goal,
            # "place": path_place,
            "return": path_return
        }

    def force_down_until_contact(self, timeout=2, contact_threshold=2):
        print("力控朝下，等待接触...")
        task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [0, 0, 1, 0, 0, 0]
        z_force = [0, 0, -5, 0, 0, 0]
        force_type = 2
        limits = [20, 20, 20, 1, 1, 1]
        self.robot_real.rtde_c.zeroFtSensor()
        self.robot_real.check_rtder_is_connected()
        fz_total = 0
        for i in range(5):
            fz_total += abs(self.robot_real.rtde_r.getActualTCPForce()[2])
        fz_start = fz_total / 5
        print(f"fz_start: {fz_start}")
        self.robot_real.rtde_c.forceMode(task_frame, selection_vector, z_force, force_type, limits)
        start_time = time.time()

        while True:
            fz = self.robot_real.rtde_r.getActualTCPForce()[2]
            print(abs(fz))
            if abs(fz) - fz_start > contact_threshold:
                print(f"接触成功！检测到z轴力: {fz:.2f} N")
                self.robot_real.rtde_c.zeroFtSensor()
                self.robot_real.rtde_c.forceMode(task_frame, [0] * 6, [0] * 6, force_type, limits)
                self.robot_real.rtde_c.forceModeStop()
                pose = self.robot_real.rtde_r.getActualTCPPose()
                return pose  # 返回接触点位姿
            if time.time() - start_time > timeout:
                print("超时未接触到力，力控退出")
                self.robot_real.rtde_c.zeroFtSensor()
                self.robot_real.rtde_c.forceMode(task_frame, [0] * 6, [0] * 6, force_type, limits)
                self.robot_real.rtde_c.forceModeStop()
                return None
            time.sleep(0.05)

    def spiral_search(self, start_pose, spiral_radius=0.004, spiral_pitch=0.002, max_turns=8, step_count=120,
                      search_timeout=6, z_disp_threshold=0.002):
        self.robot_real.check_rtder_is_connected()
        print("开始阿基米德螺旋搜孔...")
        t0 = time.time()
        status = True
        # 初始位置坐标
        x0, y0, z0 = start_pose[0:3]
        task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [0, 0, 1, 0, 0, 0]
        z_force = [0, 0, -5, 0, 0, 0]
        force_type = 2
        limits = [20, 20, 20, 1, 1, 1]
        self.robot_real.rtde_c.zeroFtSensor()
        self.robot_real.rtde_c.forceMode(task_frame, selection_vector, z_force, force_type, limits)
        for i in range(step_count):
            angle = i * (2 * np.pi / step_count) * max_turns
            r = spiral_radius + spiral_pitch * angle / (2 * np.pi)
            x = x0 + r * np.cos(angle)
            y = y0 + r * np.sin(angle)
            pose = list(start_pose)
            pose[0], pose[1] = x, y
            self.robot_real.moveL(pose, 0.01, 0.05)
            time.sleep(0.5)
            self.robot_real.check_rtder_is_connected()
            current_pose = self.robot_real.rtde_r.getActualTCPPose()
            dz = abs(current_pose[2] - z0)  # z方向的位移
            if dz > z_disp_threshold:
                print("找到孔！Z轴下移量: %.4f m" % dz)
                break
            if time.time() - t0 > search_timeout:
                print("搜孔超时！")
                status = False
                break
        self.robot_real.rtde_c.zeroFtSensor()
        self.robot_real.rtde_c.forceMode(task_frame, [0] * 6, [0] * 6, force_type, limits)
        self.robot_real.rtde_c.forceModeStop()
        return current_pose, status

    def screw_insert(self, insert_depth=0.013, total_angle=2 * np.pi, steps=50, speed=0.015):
        print("插孔旋转同步进行...")
        self.robot_real.check_rtder_is_connected()
        pose = self.robot_real.rtde_r.getActualTCPPose()
        start_pose = np.array(pose)
        start_z = start_pose[2]
        start_rotvec = start_pose[3:6].copy()   # 轴角

        for i in range(1, steps + 1):
            dz = insert_depth * i / steps
            dtheta = total_angle * i / steps
            curr_pose = start_pose.copy()
            curr_pose[2] = start_z - dz
            rot = R.from_rotvec(start_rotvec)   # 把轴角转化为一个 Rotation 对象
            delta_rot = R.from_rotvec([0, 0, dtheta])   # 绕z轴旋转的一个 Rotation 对象
            # delta_rot = rm.rotmat_from_axangle([0, 0, 1], dtheta)   # 也是绕z轴旋转的旋转矩阵
            new_rot = delta_rot * rot   # 将rot绕z轴旋转dtheta后的一个 Rotation 对象
            curr_pose[3:6] = new_rot.as_rotvec()    # 把Rotation 对象转换为轴角
            self.robot_real.moveL(curr_pose.tolist(), speed, 0.05)
            time.sleep(0.08)
        print(f"同步插入+转动，插深{insert_depth}m，旋转{np.degrees(total_angle):.1f}°，分{steps}步完成！")

    def force_hole_search_procedure(self, target_z, z_tol=0.001):
        # 1. 下压力控直到接触
        contact_pose = self.force_down_until_contact(timeout=120, contact_threshold=3)
        if contact_pose is None:
            print("未接触到工件，流程终止")
            return
        actual_z = contact_pose[2]
        dz = abs(actual_z - target_z)
        print(f"接触后Z={actual_z:.4f}，目标Z={target_z:.4f}，差值dz={dz:.6f}")
        if dz < z_tol:
            print(f"实际z与目标z差值({dz:.4f}m)小于阈值({z_tol}m)，无需搜索和插孔")
            return
        # 2. 螺旋搜孔
        end_pose, status = self.spiral_search(contact_pose, spiral_radius=0.004, spiral_pitch=0.0025,
                                              max_turns=8, step_count=200, search_timeout=120, z_disp_threshold=0.001)
        if status:
            # 3. 螺旋插入
            self.screw_insert(insert_depth=0.013, total_angle=np.pi/3*2, steps=10, speed=0.15)
        print("力控找孔+旋入流程完成 √")

    def execute_pick_place(self, target_pose, pick_path_dict, place_path_dict):
        if self.sim:
            return
        target_pos, target_rot = target_pose[:3, 3], target_pose[:3, :3]
        approach_pos = target_pos - self.config["approach_dist"] * target_rot[:, 2]
        app_conf = self._safe_ik(approach_pos, target_rot)
        path1 = copy.deepcopy(pick_path_dict['app'])
        path1.extend(pick_path_dict['grasp'])
        # init
        self.robot_real.move_jnts(jnt_values=self.robot_sim.arm_homeconf, vel=self.high_speed, acc=self.acc)
        self.robot_real.open_gripper_dh(speed=self.grip_speed, force=self.grip_force)
        # pick
        self.robot_real.move_jntspace_path(path1, toppra_vels=self.high_toppra, toppra_accs=self.toppra_accs)
        self.robot_real.close_gripper_dh(speed=self.grip_speed, force=self.grip_force)
        self.robot_real.move_jntspace_path(pick_path_dict['pick'], toppra_vels=self.high_toppra,
                                      toppra_accs=self.toppra_accs)
        # place
        self.robot_real.move_jntspace_path(place_path_dict['app'], toppra_vels=self.high_toppra,
                                      toppra_accs=self.toppra_accs)
        # searching
        self.force_hole_search_procedure(target_z=target_pos[-1])
        self.robot_real.open_gripper_dh(speed=self.grip_speed, force=self.grip_force)

        # return
        self.robot_real.move_jnts(app_conf, self.high_speed, self.acc)
        self.robot_real.move_jntspace_path(place_path_dict['return'], toppra_vels=self.high_toppra,
                                      toppra_accs=self.toppra_accs)
        print("Pick and place execution completed!")


if __name__ == "__main__":
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gripper_s = hnd.Dh50()
    rbt_s = rbt.UR7E()
    # rbt_r = ur7e_con.UR5Ag95X_RTDE(robot_ip="192.168.125.30")
    rbt_r = None

    pickplace = RobotPlanner(rbt_s, rbt_r, None, gripper_s, CONFIG, base, sim=True)
    grasp_planner = GraspPlanner(obj_name='M16_50', gripper_sim=gripper_s)
    object_trans_mat = np.array([[ 9.73457787e-01,  2.28568478e-01,  1.16785255e-02,  6.26924976e-01],
                         [ 2.28545355e-01, -9.73527688e-01,  3.29554454e-03, -1.22272822e-01],
                         [ 1.21226255e-02, -5.39000740e-04, -9.99926373e-01,  6.15309072e-02],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    target_trans_mat = np.array([[ 9.73457787e-01,  2.28568478e-01,  1.16785255e-02,  0.61336345],
                         [ 2.28545355e-01, -9.73527688e-01,  3.29554454e-03, 0.03754109],
                         [ 1.21226255e-02, -5.39000740e-04, -9.99926373e-01,  0.0615],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    pick_path, place_path, grasp_info = pickplace.plan_pick_place(object_trans_mat, target_trans_mat, grasp_planner)
    if pick_path is None or place_path is None:
        print("No valid path found.")
    else:
        pickplace.execute_pick_place(target_trans_mat, pick_path, place_path)

    base.run()

