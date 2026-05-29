import time
from typing import Literal
import robot_sim.end_effectors.gripper.dh50longfinger.dh50longfinger as dh
import robot_sim.robots.ur7e.ur7ewithoutmachine as ur7
import basis.robot_math as rm
import modeling.collision_model as cm
import numpy as np
import motion.probabilistic.rrt_connect as rrtc
from config import CONFIG_U1, CONFIG_U625, CONFIG_U1_cc_xipan, CONFIG_U625_cc_xipan
import math
from grasp_planner import Grasp_Planner
from scipy.spatial.transform import Rotation as Rot

# 抓取U625路径规划
class Path_plan(object):
    def __init__(self, rbt_s, obstacle_list, base):
        self.rbt_s = rbt_s  # 由主函数传入仿真机器人
        self.rrt_planner = rrtc.RRTConnect(self.rbt_s)
        self.component_name = 'arm'
        self.obstacle_list = obstacle_list
        self.base = base
        self.gripper = dh.Dh50()
        self.grasp_planner = Grasp_Planner(self.base,self.rbt_s,self.gripper,self.obstacle_list)
    ################U625路径规划#####################

    def U625_rot_get_min_grasp_angle_from_list(self, grasp_list, basis_rot_0):
        """
            计算出不碰撞里的最小的旋转角，对应的位置，姿态
            输入：
                grasp_list:抓取列表
                basis_rot:基准姿态
            输出：
                jaw_center_pos：抓取位置
                jaw_center_rotmat：抓取旋转矩阵
                min_angle：绕世界坐标系的z轴的最小旋转角
        """
        basis_rot = np.asarray(basis_rot_0)
        if len(grasp_list) == 0:
            print("grasp_list为空")
            return None,None,None
        seed = CONFIG_U625['grasp']['box_center_conf']
        for jaw_center_pos,jaw_center_rotmat in grasp_list:
            # ik求解
            grasp_conf = self._safe_ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat,seed = seed, method = 'tracik')
            if grasp_conf is not None:
                grasp_rot_z = jaw_center_rotmat @ basis_rot.T
                r = Rot.from_matrix(grasp_rot_z)
                min_angle = r.as_euler('xyz')[2]  # 从basis_rot到jaw_center_rotmat绕z轴旋转的弧度
                return jaw_center_pos,jaw_center_rotmat, min_angle
        print("U625ik求解都失败")
        return None,None,None

    @staticmethod
    def U625_get_high_grasp_pos(grasp_pos, d, offset):
        """
        根据U625抓取点求出抓取点上方的接近点
        输入：
            grasp_pos:抓取点位置
            d:边缘阈值距离
            offset:接近点偏移距离
        输出：
            high_grasp_pos:上方高处接近点位置
        """
        x, y, z = grasp_pos
        zz = min(z + 0.15, CONFIG_U625['grasp']['box_center_pos'][-1])  # 离抓取点的高度,不超过箱子中间高位位置的高度
        x1, x2 = CONFIG_U625['grasp']['box_pos'][0] - 0.6 / 2, CONFIG_U625['grasp']['box_pos'][0] + 0.6 / 2  # 箱子的x范围
        y1, y2 = CONFIG_U625['grasp']['box_pos'][1] - 0.4 / 2, CONFIG_U625['grasp']['box_pos'][1] + 0.4 / 2  # 箱子的y范围
        if x < x1 + d:
            xx = x + offset
            if y < y1 + d:
                yy = y + offset
                print("U625在左上")
            elif y < y2 - d:
                yy = y
                print("U625在上")
            else:
                yy = y - offset
                print("U625在右上")
        elif x < x2 - d:
            xx = x
            if y < y1 + d:
                yy = y + offset
                print("U625在左")
            elif y < y2 - d:
                yy = y
                print("U625在中间")
            else:
                yy = y - offset
                print("U625在右")
        else:
            xx = x-offset
            if y < y1 + d:
                yy = y+offset
                print("U625在左下")
            elif y < y2 - d:
                yy = y
                print("U625在下")
            else:
                yy = y-offset
                print("U625在右下")
        high_grasp_pos = np.array([xx, yy, zz])
        return high_grasp_pos

    def U625_grasp_and_place_path(self, current_conf, pos):
        '''
            计算U625的抓取和放置路径
            输入：
                pos：物体位置
            输出：
                dict：接近路径和放置路径
        '''
        start_time = time.time()
        # 物体姿态
        rot = np.eye(3)
        # 根据物体位置和姿态计算出抓取信息列表
        grasp_list = self.grasp_planner.U625_to_jaw(pos,rot)
        if len(grasp_list) == 0:
            print("U625抓取列表为空，抓取姿态都碰撞")
            # self.base.run()
            return None
        # 抓取位置，姿态和旋转角度，从4个候选姿态中选出
        grasp_pos, grasp_rot, grasp_angle = self.U625_rot_get_min_grasp_angle_from_list(grasp_list, CONFIG_U1['grasp']['wait_rot'])
        if grasp_pos is None:
            print("U625抓取位姿计算失败")
            # self.base.run()
            return None
        # 抓取位姿
        grasp_pose = self.rbt_s.get_real_tcp_pose(grasp_pos,grasp_rot)
        print(f"U625grasp_pose:{grasp_pose}")
        # 抓取低位
        low_grasp_pos = grasp_pos + np.array([0, 0, CONFIG_U625['grasp']['prepare_height']])
        low_grasp_pose = self.rbt_s.get_real_tcp_pose(low_grasp_pos,grasp_rot)
        print(f"U625low_grasp_pose:{low_grasp_pose}")
        # 抓取高位
        high_grasp_pos = self.U625_get_high_grasp_pos(grasp_pos, 0.15, 0.03)
        high_grasp_conf = self._safe_ik(high_grasp_pos, grasp_rot, seed=CONFIG_U625['grasp']['box_center_conf'], method = 'tracik')
        if high_grasp_conf is None:
            print(f"U625抓取高位计算失败，high_grasp_pos:{high_grasp_pos}grasp_rot:{grasp_rot}")
            # self.base.run()
            return None
        # 接近路径1
        # 先使用直线+旋转路径规划
        approach_path1 = self.get_line_change_rot(
            seed=current_conf, start_pos=CONFIG_U1['grasp']['wait_pos'],
            goal_pos=high_grasp_pos, start_rot=CONFIG_U1['grasp']['wait_rot'], angle=grasp_angle)
        if approach_path1 is None:
            print("U625直线接近路径1求解失败，改用rrt路径规划")
            # 直线碰撞再使用rrt路径规划
            approach_path1 = self.plan_path(current_conf, high_grasp_conf, 0.07)
            if approach_path1 is None:
                print("U625接近路径1计算失败")
                # self.base.run()
                return None
        # 接近路径2，直线路径
        approach_path2 = self.get_line_path(approach_path1[-1],high_grasp_pos,low_grasp_pos,grasp_rot)
        if approach_path2 is None:
            print("U625接近路径1计算失败")
            # self.base.run()
            return None
        # 合并中间平滑处理,最终合并路径
        approach_path = self.get_smooth_path(approach_path1+ approach_path2,0.05)
        # approach_path = self.smooth_two_path(approach_path1, approach_path2,0.1)
        # 放置路径1，从抓取低点到箱子中心高点
        place_path1 = self.get_line_change_rot(seed=CONFIG_U625['grasp']['box_center_conf'], start_pos=low_grasp_pos, goal_pos=CONFIG_U625['grasp']['box_center_pos'],
                                               start_rot=grasp_rot , angle= -grasp_angle)
        if place_path1 is None:
            print("U625放置路径1计算失败")
            # self.base.run()
            return None
        # 中间平滑处理,最终合并路径
        place_path = self.smooth_two_path(place_path1, CONFIG_U625['place']['place_path'],0.05)
        dict = {
            'app': approach_path,   # 从等待点到抓取低点
            'low_grasp_pose':low_grasp_pose,    # 抓取低点位姿
            'grasp_pose': grasp_pose,   # 抓取位姿
            'place': place_path,    # 从抓取低点到放置高点
        }
        print(f"U625路径计算总时间{time.time()-start_time}")
        return dict

    #################U1路径规划###################
    def plan_path(self, current_conf, target_conf, step = 0.07):  # 从当前位置移动到目标位置
        """
        :param current_conf: 起始关节角
        :param target_conf: 目标关节角
        """
        path_list = self.rrt_planner.plan(
            component_name=self.component_name,
            start_conf=current_conf,
            goal_conf=target_conf,
            obstacle_list=self.obstacle_list,
            otherrobot_list=[],
            ext_dist=step,   # 扩展步长
            max_iter=300,   # 最大迭代次数
            max_time=15.0,  # 最大规划时间（秒）
            smoothing_iterations=50,    # 平滑迭代次数
            animation=False)
        if path_list is None:
            # 检查首末点是否碰撞
            self.rbt_s.fk('arm',current_conf)
            if self.rrt_planner._is_collided(self.component_name, current_conf, self.obstacle_list, []):
                print(f"RRT开始点 {list(current_conf)} 碰撞!")
                return None
            self.rbt_s.fk('arm', target_conf)
            if self.rrt_planner._is_collided(self.component_name, target_conf, self.obstacle_list, []):
                print(f"RRT结束点 {list(target_conf)} 碰撞!")
                return None
            print("RRT路径规划失败")
            # 把步长改大
            path_list = self.rrt_planner.plan(
                component_name=self.component_name,
                start_conf=current_conf,
                goal_conf=target_conf,
                obstacle_list=self.obstacle_list,
                otherrobot_list=[],
                ext_dist=min(step+0.05,0.11),  # 扩展步长,最大0.11
                max_iter=300,  # 最大迭代次数
                max_time=15.0,  # 最大规划时间（秒）
                smoothing_iterations=50,  # 平滑迭代次数
                animation=False)
            if path_list is None:
                print("步长改大RRT还是规划失败")
                # 把步长改小
                path_list = self.rrt_planner.plan(
                    component_name=self.component_name,
                    start_conf=current_conf,
                    goal_conf=target_conf,
                    obstacle_list=self.obstacle_list,
                    otherrobot_list=[],
                    ext_dist=max(step - 0.05, 0.04),  # 扩展步长,最小0.04
                    max_iter=300,  # 最大迭代次数
                    max_time=15.0,  # 最大规划时间（秒）
                    smoothing_iterations=50,  # 平滑迭代次数
                    animation=False)
                if path_list is None:
                    print("步长改小RRT还是规划失败")

        return path_list

    def get_line_path(self, seed=None, start=None, goal=None, rot=None): # 直线路径规划,返回关节点路径
        jnts = []
        cur_seed = seed  # 更新种子，提高连续性
        # 计算起点到终点的距离
        distance = np.linalg.norm(np.array(goal) - np.array(start))
        # 根据距离和步长确定点数（至少 10 点）
        num = max(10, int(distance * 30))  # type: ignore
        for pos in np.linspace(start, goal, num):
            conf = self._safe_ik(pos, rot, seed = cur_seed, method='tracik')
            if conf is None:    # ik求解失败
                print(f"机器人tracik求解位置{pos}旋转{rot}关节seed{cur_seed}碰撞，直线路径求解失败")
                return None
            else:
                jnts.append(conf)
                cur_seed = conf  # 更新种子，提高连续性
        return jnts

    def get_line_change_rot(self,seed=None, start_pos=None, goal_pos=None, start_rot=None , angle=0):
        '''
        从起始到终点的直线路径，但姿态也会绕z轴旋转angle角度
        输入:
            seed:初始关节角种子
            start_pos:开始位置
            goal_pos:终点位置
            start_rot:初始姿态
            angle:绕世界坐标系z轴旋转的角度
        输出:
            jnts:关节路径点
        '''
        if angle == 0:  # 规划直线路径
            return self.get_line_path(seed=seed, start=start_pos, goal=goal_pos, rot=start_rot)
        # 规划直线+旋转路径
        jnts = []
        cur_seed = seed  # 更新种子，提高连续性
        # 计算起点到终点的距离
        distance = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
        # 根据距离和步长确定点数（至少 10 点，最长 cm一个点）
        num = max(10, int(distance * 30)) # type:ignore
        pos_change = (goal_pos - start_pos) / num
        angle_change = angle/num
        cur_pos = start_pos
        cur_angle = 0
        cur_conf = self._safe_ik(cur_pos, start_rot, seed=cur_seed, method='tracik')    # 起点关节点
        if cur_conf is not None:
            jnts.append(cur_conf)
            cur_seed = cur_conf  # 更新关节点种子
        else:
            print(f"直线旋转路径起点关节点计算失败，start_pos:{start_pos},start_rot:{start_rot},seed:{cur_seed}")
            return None
        for i in range(num):    # 总共num次循环
            cur_pos += pos_change
            cur_angle += angle_change
            cur_rot = rm.rotmat_from_axangle([0, 0, 1], cur_angle) @ start_rot
            cur_conf = self._safe_ik(cur_pos, cur_rot, seed=cur_seed, method='tracik')
            if cur_conf is not None:
                jnts.append(cur_conf)
                cur_seed = cur_conf # 更新关节点种子
            else:
                print(f"直线旋转路径关节点计算失败，pos:{cur_pos},cur_rot:{cur_rot},cur_seed:{cur_seed}")
                return None
        return jnts

    def _safe_ik(self, tgt_pos, tgt_rotmat, seed=None, method: Literal['ik', 'tracik'] = 'tracik'):
        seed0 = CONFIG_U1['grasp']['wait_conf'] if seed is None else seed.copy()
        # 候选种子
        seed1 = CONFIG_U1['grasp']['wait_conf']
        seed2 = CONFIG_U1['grasp']['box_center_conf']
        seed3 = CONFIG_U625['grasp']['box_center_conf']

        if method == 'ik':
            conf = self.rbt_s.ik("arm", tgt_pos, tgt_rotmat, seed_jnt_values=seed0)
        # tracik求解
        elif method == 'tracik':
            conf = self.rbt_s.tracik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed0, solver_type='Distance')
        else:
            raise ValueError("_safe_ik Wrong method")
        if conf is None:
            for seed in [seed1, seed2, seed3]:
                conf = self.rbt_s.tracik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed, solver_type='Distance')
                if conf is not None:
                    self.rbt_s.fk("arm", conf)
                    if not self.rbt_s.is_collided(self.obstacle_list):  # 无碰撞
                        return conf
                    else:
                        print(f"机器人求解位置{tgt_pos}旋转{tgt_rotmat}种子{seed}关节{conf}碰撞")
            else:
                print("3种seed的ik求解都失败")
                return None
        # 碰撞检测
        if conf is not None:
            self.rbt_s.fk("arm", conf)
            if self.rbt_s.is_collided(self.obstacle_list):
                print(f"机器人{method}求解位置{tgt_pos}旋转{tgt_rotmat}种子{seed0}关节{conf}碰撞")
                self.rbt_s.gen_meshmodel(rgba=[1,0,0,0.3]).attach_to(self.base)
                return None
        return conf

    @staticmethod
    def get_high_grasp_pos(grasp_pos, d, offset):
        """
        根据抓取点求出抓取点上方的接近点
        输入：
            grasp_pos:抓取点位置
            d:边缘阈值距离
            offset:接近点偏移距离
        输出：
            high_grasp_pos:上方高处接近点位置
        """
        x, y, z = grasp_pos
        zz = min(z + 0.15, CONFIG_U1['grasp']['box_center_pos'][-1])  # 离抓取点的高度,不超过箱子中间高位位置的高度
        x1, x2 = CONFIG_U1['grasp']['box_pos'][0] - 0.365/2, CONFIG_U1['grasp']['box_pos'][0] + 0.365/2 # 箱子的x范围
        y1, y2 = CONFIG_U1['grasp']['box_pos'][1] - 0.565/2, CONFIG_U1['grasp']['box_pos'][1] + 0.565/2 # 箱子的y范围
        if x < x1 + d:
            xx = x + offset
            if y < y1 + d:
                print("在左上角")
                yy =  y + offset
            elif y < y2 - d:
                print("在上边")
                yy = y
            else:
                print("在右上角")
                yy = y - offset
        elif x < x2 - d:
            xx = x
            if y < y1 + d:
                print("在左边")
                yy = y + offset
            elif y < y2 - d:
                print("在中间")
                yy = y
            else:
                print("在右边")
                yy = y - offset
        else:
            xx = x - offset
            if y < y1 + d:
                print("在左下角")
                yy = y + offset
            elif y < y2 - d:
                print("在下边")
                yy = y
            else:
                print("在右下角")
                yy = y - offset
        high_grasp_pos = np.array([xx, yy, zz])
        return high_grasp_pos

    def U1_get_min_grasp_angle_from_list(self, grasp_list,wait_rot_0):
        """
        得到与U1箱子不碰撞的最小的抓取角
        输入：
            grasp_list:抓取列表
            wait_rot:基准姿态
        输出：

            min_grasp_angle:最小的抓取角
            min_grasp_angle:最小的抓取角时的姿态
        """
        wait_rot = np.asarray(wait_rot_0)
        if len(grasp_list) == 0:
            print("grasp_list长度为0")
            return None, None, None
        angle_list = [] # 旋转角度和抓取姿态列表
        for i in range(len(grasp_list)):    # i从0到len(grasp_list)-1
            jaw_center_pos, jaw_center_rotmat = grasp_list[i]
            # ik求解
            grasp_conf_0 = self._safe_ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat,
                                       seed=CONFIG_U1['grasp']['box_center_conf'],
                                       method='tracik')
            if grasp_conf_0 is not None:
                # print(f"U1第{i}个抓取ik求解成功")
                grasp_rot_z = jaw_center_rotmat @ wait_rot.T
                r = Rot.from_matrix(grasp_rot_z)
                z_angle_0 = r.as_euler('xyz')[2]  # 从wait_rot到jaw_center_rotmat绕z轴旋转的弧度
                # 比较z_angle,z_angle+np.pi,和z_angle-np.pi的绝对值大小，计算出当前姿态的旋转最小角
                min_abs_angle = min(abs(z_angle_0-np.pi), abs(z_angle_0), abs(z_angle_0+np.pi))
                if min_abs_angle == abs(z_angle_0-np.pi):
                    min_angle = z_angle_0-np.pi
                    grasp_rot_1 = rm.rotmat_from_axangle([0, 0, 1], min_angle) @ wait_rot
                    grasp_conf_1 = self._safe_ik(tgt_pos=jaw_center_pos, tgt_rotmat=grasp_rot_1,
                                               seed=CONFIG_U1['grasp']['box_center_conf'],
                                               method='tracik')
                elif min_abs_angle == abs(z_angle_0):
                    min_angle = z_angle_0
                    grasp_rot_1 = jaw_center_rotmat
                    grasp_conf_1 = grasp_conf_0
                else:
                    min_angle = z_angle_0+np.pi
                    grasp_rot_1 = rm.rotmat_from_axangle([0, 0, 1], min_angle) @ wait_rot
                    grasp_conf_1 = self._safe_ik(tgt_pos=jaw_center_pos,
                                               tgt_rotmat=grasp_rot_1,
                                               seed=CONFIG_U1['grasp']['box_center_conf'],
                                               method='tracik')
                if grasp_conf_1 is not None:
                    angle_list.append([min_angle, jaw_center_pos, grasp_rot_1])
                else:
                    angle_list.append([z_angle_0, jaw_center_pos, jaw_center_rotmat])

        if len(angle_list) == 0:
            return None, None, None
        min_angle_z, min_grasp_pos, min_grasp_rot = angle_list[0]
        for angle, grasp_pos, grasp_rot in angle_list[1:]:
            if abs(angle) < abs(min_angle_z):
                min_angle_z = angle
                min_grasp_pos = grasp_pos
                min_grasp_rot = grasp_rot

        return min_angle_z, min_grasp_pos, min_grasp_rot

    # 平滑路径处理
    def get_smooth_path(self, path, step = 0.1):
        return self.rrt_planner._smooth_path(
                     'arm', path,
                     obstacle_list=self.obstacle_list,
                     otherrobot_list=[],
                     granularity=step,  # 步长
                     iterations=50,
                     animation=False)

    def smooth_two_path(self, path1, path2, step = 0.1):
        if path1 is None:
            print("平滑路径1为None")
            return None
        if path2 is None:
            print("平滑路径2为None")
            return None
        # 中间1/3部分进行平滑
        smooth_path_0 = path1[int(len(path1)*2/3):] + path2[0:int(len(path2)/3)]
        smooth_path = self.get_smooth_path(smooth_path_0,step)
        # 合并最终路径
        path = path1[0:int(len(path1)*2/3)] + smooth_path + path2[int(len(path2)/3):]
        return path

    @staticmethod
    def get_min_place_angle(angle):  # 求最小旋转的放置角
        min_place_angle = min(
            abs(angle),
            abs(angle + np.pi), abs(angle - np.pi),
        )
        if min_place_angle == abs(angle):
            min_angle = angle
        elif min_place_angle == abs(angle + np.pi):
            min_angle = angle + np.pi
        else:
            min_angle = angle - np.pi
        return min_angle

    def U1_grasp_and_place_path(self,cls_id, pos, angle, wait_rot):
        """
        计算U1的抓取路径
        输入:
            cls_id:物体类别
            pos:抓取位置
            angle:物体角度
            wait_rot:基准姿态
        输出:
            conf_dict:抓取和放置的路径
        """
        start_time = time.time()
        seed_jnts = CONFIG_U1["grasp"]['box_center_conf']
        # 物体旋转矩阵(先倒放，再绕世界坐标系的z轴旋转angle角度)
        U1_rot = rm.rotmat_from_axangle([0, 0, 1], np.pi+ angle) @ rm.rotmat_from_axangle([1, 0, 0], np.pi)
        # 根据物体位置和旋转矩阵计算抓取点的位置和旋转矩阵
        grasp_info_list = self.grasp_planner.U1_to_jaw(pos, U1_rot)
        if len(grasp_info_list) == 0:
            print("U1所有抓取点都碰撞")
            # self.base.run()
            return None
        # 找到旋转角最小的抓取角和抓取位置，旋转矩阵
        grasp_angle, grasp_pos, grasp_rot = self.U1_get_min_grasp_angle_from_list(grasp_info_list,wait_rot)
        if grasp_angle is None:
            print("U1抓取最小角度计算失败")
            # self.base.run()
            return None
        print(f"U1抓取角度：{grasp_angle}")
        # 抓取位姿
        grasp_pose = self.rbt_s.get_real_tcp_pose(grasp_pos, grasp_rot)
        print("U1grasp_pose", grasp_pose)
        # print(f"计算最小角度和姿态时间:{time.time()}")
        # 抓取点上方高处位置
        high_grasp_pos = self.get_high_grasp_pos(grasp_pos, 0.10, 0.02)
        high_grasp_conf = self._safe_ik(tgt_pos=high_grasp_pos, tgt_rotmat=grasp_rot, seed=seed_jnts, method='tracik')
        if high_grasp_conf is None:
            print(f"U1抓取高处ik计算失败:high_grasp_pos：{high_grasp_pos}grasp_rot：{grasp_rot}，seed：{seed_jnts}")
            # self.base.run()
            return None
        print(f"high_grasp_conf:{high_grasp_conf}")
        # print(f"计算高处ik时间:{time.time()}")
        # 接近路径（速度快）
        # 先使用直线+旋转路径规划
        approach_path1 = self.get_line_change_rot(
            seed=CONFIG_U1['grasp']['wait_conf'], start_pos=CONFIG_U1['grasp']['wait_pos'],
            goal_pos=high_grasp_pos, start_rot=CONFIG_U1['grasp']['wait_rot'], angle=grasp_angle)
        if approach_path1 is None:
            print("U1直线接近路径1求解失败，改用rrt路径规划")
            # 直线碰撞再使用rrt路径规划
            approach_path1 = self.plan_path(CONFIG_U1['grasp']['wait_conf'],high_grasp_conf,0.07)
            if approach_path1 is None:
                print("U1接近路径1计算失败")
                # self.base.run()
                return None
        # print(f"计算接近路径1时间:{time.time()}")
        # 抓取低点位置
        low_grasp_pos = grasp_pos+np.array([0,0,CONFIG_U1['grasp']['prepare_height']])
        # 抓取低点位姿
        low_grasp_pose = self.rbt_s.get_real_tcp_pose(low_grasp_pos, grasp_rot)
        print(f"U1low_grasp_pose:{low_grasp_pose}")
        # 从抓取高点到抓取低点
        approach_path2 = self.get_line_path(approach_path1[-1], high_grasp_pos, low_grasp_pos, grasp_rot)  # 可能有斜移
        if approach_path2 is None:
            print("U1接近路径2计算失败")
            # self.base.run()
            return None
        # print(f"计算接近路径2时间:{time.time()}")
        # 合并中间平滑处理,最终合并路径
        # approach_path = self.get_smooth_path(approach_path1+ approach_path2, 0.05)
        approach_path = self.smooth_two_path(approach_path1, approach_path2,0.1)
        print(f"U1approach_path:{approach_path}")
        # print(f"计算抓取路径时间:{time.time()}")
        # print(f"接近路径：{approach_path}")
        if cls_id == 1: # 正放物体,也要计算放置路径，从抓取低点到抛弃点
            # 从抓取低点到抓取高点
            place_path1 = approach_path2[::-1]
            # 计算直线旋转路径,从抓取高点到抛弃点
            place_path2 = self.get_line_change_rot(
                seed=place_path1[-1], start_pos=high_grasp_pos,
                goal_pos=CONFIG_U1['abandon']['abandon_pos'],start_rot=grasp_rot, angle=-grasp_angle
            )
            if place_path2 is None:
                print("U1丢弃路径2计算失败")
                # self.base.run()
                return None
            # 中间平滑处理,合并最终放置路径，从抓取低点到等待点
            place_path = self.smooth_two_path(place_path1, place_path2, step=0.1)
            place_pose,high_place_pose = None,None
        else:   # 倒放物体
            # 放置角度
            place_angle = grasp_angle - angle  # 放置角度 = 抓取角度-物体角度，范围为0,np.pi,-np.pi,np.pi/3,-np.pi/3,np.pi*2/3,-np.pi*2/3
            place_angle_1 = self.get_min_place_angle(place_angle)   # 范围为0，np.pi/3，-np.pi/3，
            print(f"U1放置角度:{place_angle_1}")
            # 放置高点和放置点
            if abs(place_angle_1) <= 0.1:
                high_place_pose = CONFIG_U1['place']['0']['high_place_pose']
                place_pose = CONFIG_U1['place']['0']['place_pose']
                place_path2 = CONFIG_U1['place']['0']['place_path']
            elif abs(place_angle_1-np.pi/3) < 0.1:
                high_place_pose = CONFIG_U1['place']['60']['high_place_pose']
                place_pose = CONFIG_U1['place']['60']['place_pose']
                place_path2 = CONFIG_U1['place']['60']['place_path']
            else:
                high_place_pose = CONFIG_U1['place']['-60']['high_place_pose']
                place_pose = CONFIG_U1['place']['-60']['place_pose']
                place_path2 = CONFIG_U1['place']['-60']['place_path']
            # 放置路径1，从抓取低点到箱子中心高点，直线旋转路径
            place_path1 = self.get_line_change_rot(seed=approach_path[-1], start_pos=low_grasp_pos, goal_pos=CONFIG_U1['grasp']['box_center_pos'],
                                                 start_rot=grasp_rot, angle=-grasp_angle)   # 绕z轴旋转-grasp_angle弧度转回去
            if place_path2 is None:
                print("U1放置路径2计算失败")
                # self.base.run()
                return None
            # 中间平滑处理,合并最终放置路径
            place_path = self.smooth_two_path(place_path1, place_path2,step = 0.1)

        conf_dict = {
            "app": approach_path,    # 抓取起点接近到抓取低点,快速
            "low_grasp_pose": low_grasp_pose,   # 抓取低位位姿
            "grasp_pose": grasp_pose,   # 抓取位姿
            "place":place_path,    # 从抓取低点到放置高点
            "high_place_pose":high_place_pose,  # 放置高点位姿
            "place_pose":place_pose,    # 放置点位姿
        }
        # print(f"U1place_path:{place_path}")
        print(f"U1抓取和放置路径规划总时间：{time.time() - start_time}")
        return conf_dict


if __name__ == '__main__':
    import visualization.panda.world as wd  # 仿真环境
    import modeling.geometric_model as gm
    import robot_sim.robots.ur7e.ur7ewithoutmachine as cbt

    # 1. 仿真环境
    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    rbt_s = cbt.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                     enable_cc=True)  # 仿真机器人
    # rbt_s.gen_meshmodel().attach_to(base)
    # 2. 障碍物模型
    obstacle_list = rbt_s.get_obstacle_list(base,False)
    Path_plan = Path_plan(rbt_s=rbt_s, obstacle_list=obstacle_list,base=base)
    # current_conf = np.array([0.46579   ,  -1.7793   ,   1.7024    , -1.4939  ,   -1.5708   ,   -1.105])
    # grasp_pos = np.array([0.90129  ,  -0.37715  ,   0.88578])
    # grasp_rot = [[   -1         ,  0 , 1.2246e-16],
    #              [          0       ,    1        ,   0],
    #              [-1.2246e-16      ,     0      ,    -1]]
    # path = Path_plan.U625_grasp_and_place_path(current_conf, grasp_pos, grasp_rot)
    # place_path = path['place']
    # print(place_path)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in place_path:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0,1,0,0.5])
    #     rbt_mesh.attach_to(base)
    # base.run()

    # wait_conf = CONFIG_U1['grasp']['wait_conf']
    # wait_pos = CONFIG_U1['grasp']['wait_pos']
    # wait_rot = CONFIG_U1['grasp']['wait_rot']
    # goal_pos = CONFIG_U1['place']['60']['high_place_pos']
    # start_time = time.time()
    # path = Path_plan.get_line_change_rot(seed=wait_conf, start_pos=wait_pos, goal_pos=goal_pos, start_rot=wait_rot, angle=np.pi/3)
    # print(path)
    # print(f"用时{time.time() - start_time}")
    # rbt_s.fk("arm", path[-1])  ## 用正运动学（fk）更新机器人姿态
    # rbt_s.gen_meshmodel(rgba=[0.5, 0.5, 0.5, 1]).attach_to(base)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in path:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0,1,0,0.5])
    #     rbt_mesh.attach_to(base)
    # base.run()

    # U1路径规划
    cls_id = 0
    pos = CONFIG_U1['grasp']['box_pos']+np.array([0.14,-0.16,0.05])  # 物体位置
    angle = np.pi/2 # 物体相对于世界坐标系z轴的旋转角度

    wait_rot = CONFIG_U1['grasp']['wait_rot']
    grasp_and_place_path = Path_plan.U1_grasp_and_place_path(cls_id, pos, angle, wait_rot)
    app_path = grasp_and_place_path['app']
    place_path = grasp_and_place_path['place']
    # 展示路径
    rbt_mesh = None
    for jnts1 in app_path:
        rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
        rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
        rbt_mesh.attach_to(base)
    for jnts1 in place_path:
        rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
        rbt_mesh = rbt_s.gen_meshmodel(rgba=[1, 1, 0, 0.5])
        rbt_mesh.attach_to(base)
    base.run()

    # # # U625路径规划
    # current_conf = CONFIG_U1['grasp']['wait_conf']
    # # U625_pos = CONFIG_U625['grasp']['box_pos']+np.array([0.1,0.13,0.05])
    # U625_pos =np.array([  1.0123  ,  -0.28532  ,   0.86421])
    # grasp_and_place_path = Path_plan.U625_grasp_and_place_path(current_conf, U625_pos)
    # app_path = grasp_and_place_path['app']
    # place_path = grasp_and_place_path['place']
    # # print(f"app_path:{app_path}")
    # # print(f"place_path:{place_path}")
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in app_path:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # rbt_mesh = None
    # for jnts1 in place_path:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[1, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # base.run()

    # 吸盘路径规划
    # # 抓取低点
    # conf0 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf0']
    # # 抓取上方点
    # conf2 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf2']
    # # 纸板上方高点
    # conf3 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf3']
    # # 纸板正上方低点
    # conf4 = CONFIG_U1_cc_xipan['vacuum_clean2']['conf4']
    # # 丢弃点
    # conf6 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf6']
    # # 丢弃中间点
    # conf7 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf7']
    # # 回来抓取高点
    # conf8 = CONFIG_U1_cc_xipan['vacuum_clean1']['conf8']

    # # 抓取低点
    # conf0 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf0']
    # # 抓取上方点
    # conf2 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf2']
    # # 纸板上方高点
    # conf3 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf3']
    # # 纸板正上方低点
    # conf4 = CONFIG_U625_cc_xipan['vacuum_clean2']['conf4']
    # pos4 = CONFIG_U625_cc_xipan['vacuum_clean2']['pos4']
    # rot4 = CONFIG_U625_cc_xipan['vacuum_clean2']['rot4']
    # # 纸板上方更高点
    # pos7 = CONFIG_U625_cc_xipan['vacuum_clean1']['pos7']
    # conf7 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf7']
    #
    # # 丢弃点
    # conf6 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf6']
    # # 回来放置高点
    # conf8 = CONFIG_U625_cc_xipan['vacuum_clean1']['conf8']

    # 从等待点到抓取低点
    # path1 = Path_plan.plan_path(CONFIG_U1['grasp']['wait_conf'], conf0)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in path1:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # print(path1)

    # # 从抓取低点到吸取低点，path2
    # path1 = Path_plan.plan_path(conf0,conf2)
    # path2 = Path_plan.plan_path(conf2,conf3, step = 0.1)
    # path3 = Path_plan.plan_path(conf3,conf4)
    # smooth1 = Path_plan.smooth_two_path(path1,path2,step=0.05)
    # smooth2 = Path_plan.smooth_two_path(smooth1,path3,step=0.05)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in smooth2:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # print(smooth2)

    # # # 从纸板低点到丢弃点
    # path0 = Path_plan.plan_path(conf4,conf7,0.05)
    # path0 = Path_plan.get_line_path(conf4, pos4, pos7, rot4)
    # path1 = Path_plan.plan_path(conf7, conf6, 0.07)
    # smooth1 = Path_plan.smooth_two_path(path0, path1, step=0.05)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in  smooth1:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # print(smooth1)

    # 从丢弃点到抓取低点
    # path1 = Path_plan.plan_path(conf6, conf8,0.1)   # 先到回来的抓取高点
    # path2 = Path_plan.plan_path(conf8, conf0)
    # smooth1 = Path_plan.smooth_two_path(path1, path2, step=0.1)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in smooth1:
    #     rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5])
    #     rbt_mesh.attach_to(base)
    # print(smooth1)

    base.run()
