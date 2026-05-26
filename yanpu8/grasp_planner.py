import os
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh50longfinger.dh50longfinger as dh
import robot_sim.robots.ur7e.ur7ewithoutmachine as UR7E
import basis.robot_math as rm
from yanpu_ur8.config import CONFIG_U1, CONFIG_U625

class Grasp_Planner(object):    # 从物体到手爪中心的坐标变换

    def __init__(self, base0, rbt_s0, gripper_s0, obstacle_list0,):
        self.base = base0    # 仿真环境
        self.rbt_s = rbt_s0  # 仿真机器人
        self.gripper_s = gripper_s0  # 仿真手爪
        self.obstacle_list = obstacle_list0  # 障碍物信息
        # 当前目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # 从pickle文件中获取抓取姿态
        self.U1_dh50_list = gpa.load_pickle_file('u',   # 必须与grasping_antipodal_planning.py文件中定义的物体名相同
                                                 root=self.current_dir,
                                               file_name='grasp/U1_dh50.pickle')  # 从pickle文件中加载预计算抓取姿态
        self.U625_dh50_list = gpa.load_pickle_file('U625',
                                              root=self.current_dir,
                                              file_name='grasp/U625_dh50.pickle')
        # 物体初始化
        U1_stl_path = os.path.join(self.current_dir, 'grasp/u.STL')
        U625_stl_path = os.path.join(self.current_dir, 'grasp/U625.STL')
        self.U1_obj = cm.CollisionModel(U1_stl_path)  # 创建碰撞模型
        self.U625_obj = cm.CollisionModel(U625_stl_path)  # 创建碰撞模型

    def U1_to_jaw(self,pos,rot):
        """
        由物体的位置和姿态计算出手爪中心的位置和姿态
        输入:
            pos:物体位置
            rot:物体旋转矩阵
        输出:
            grasp_info_list:[[grasp_pos,grasp_rot],[...]...]抓取信息列表
            grasp_pos:抓取位置
            grasp_rot:抓取旋转矩阵，未考虑绕z轴旋转180度
        """
        self.U1_obj.set_rgba([.9, .75, .35, 1])  # 颜色
        self.U1_obj.set_pos(pos)  # 位置
        self.U1_obj.set_rotmat(rot)  # 姿态
        self.U1_obj.show_localframe()   # 物体坐标系
        self.U1_obj.attach_to(self.base)
        grasp_info_list = []    # 抓取信息列表
        # 计算
        for grasp_info in self.U1_dh50_list:
            # 获取相对信息
            jaw_width0, jaw_center_pos0, jaw_center_rotmat0, hnd_pos0, hnd_rotmat0 = grasp_info
            # 计算手爪中心的位置
            jaw_center_pos1 = pos + rot @ jaw_center_pos0
            # 计算手爪中心的旋转矩阵：先是手爪中心到物体的旋转矩阵，再左乘物体到世界的旋转矩阵
            jaw_center_rotmat1 = rot @ jaw_center_rotmat0
            # 手爪移到目标位置
            self.gripper_s.grip_at_with_jcpose(jaw_center_pos1, jaw_center_rotmat1, jaw_width0)
            # 检查单独的手爪是否与碰撞体碰撞
            if not self.gripper_s.is_collided(self.obstacle_list):  # 没碰撞
                # self.gripper_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(self.base)
                grasp_info_list.append([jaw_center_pos1,jaw_center_rotmat1])
            else:   # 碰撞了
                # self.gripper_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(self.base)
                # self.gripper_s.show_cdprimit()  # 展示碰撞体
                pass
        return grasp_info_list

    def U625_to_jaw(self,pos,rot):
        """
        由物体的位置和姿态计算出手爪中心的位置和姿态
        输入:
            pos:物体位置
            rot:物体旋转矩阵
        输出:
            grasp_info_list:[[grasp_pos,grasp_rot],[...]...]抓取信息列表
            grasp_pos:抓取位置
            grasp_rot:抓取旋转矩阵，未考虑绕z轴旋转180度
        """
        self.U625_obj.set_rgba([.9, .75, .35, 1])  # 颜色
        self.U625_obj.set_pos(pos)  # 位置
        self.U625_obj.set_rotmat(rot)  # 姿态
        self.U625_obj.show_localframe()   # 物体坐标系
        self.U625_obj.attach_to(self.base)
        grasp_info_list = []    # 抓取信息列表
        # 计算
        for grasp_info in self.U625_dh50_list:
            # 获取相对信息
            jaw_width0, jaw_center_pos0, jaw_center_rotmat0, hnd_pos0, hnd_rotmat0 = grasp_info
            # 计算手爪中心的位置
            jaw_center_pos1 = pos + rot @ jaw_center_pos0
            # 计算手爪中心的旋转矩阵：先是手爪中心到物体的旋转矩阵，再左乘物体到世界的旋转矩阵
            jaw_center_rotmat1 = rot @ jaw_center_rotmat0
            # 手爪移到目标位置
            self.gripper_s.grip_at_with_jcpose(jaw_center_pos1, jaw_center_rotmat1, jaw_width0)
            # 检查单独的手爪是否与碰撞体碰撞
            if not self.gripper_s.is_collided(self.obstacle_list):  # 没碰撞
                # self.gripper_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(self.base)
                grasp_info_list.append([jaw_center_pos1,jaw_center_rotmat1])
            else:   # 碰撞了
                self.gripper_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(self.base)
                self.gripper_s.show_cdprimit()  # 展示碰撞体
                pass
        return grasp_info_list


if __name__ == '__main__':
    # 仿真环境
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)
    rbt_s = UR7E.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                   enable_cc=True)  # 仿真机器人
    gripper_s = dh.Dh50()  # 创建手爪实例
    obstacle_list = rbt_s.get_obstacle_list(base,False)
    object_jaw = Grasp_Planner(base, rbt_s, gripper_s, obstacle_list)

    # # U1初始物体
    # obj_pos = np.array([0.232, 0.32, 0.9])
    # obj_rot = rm.rotmat_from_axangle([0, 0, 1], math.pi/3) @ rm.rotmat_from_axangle([1, 0, 0], math.pi)    # wait_rot
    # # 计算手爪无碰撞的抓取中心列表
    # jaw_center_info_list = object_jaw.U1_to_jaw(obj_pos, obj_rot)
    # # base.run()
    # conf_list=[]
    # print(f"jaw_center_info_list:{jaw_center_info_list}")
    # seed = np.array([-0.057731628145258965, -1.3630699998055795, 1.542364674321553, -1.7500909358126648, -1.5707964340644267, -1.6285276902039398])
    # for jaw_center_info in jaw_center_info_list: # 不碰撞的手爪列表
    #     jaw_center_pos, jaw_center_rotmat = jaw_center_info
    #     conf = rbt_s.tracik(tgt_pos=jaw_center_pos,
    #                    tgt_rotmat=jaw_center_rotmat,
    #                    seed_jnt_values=seed,
    #                     solver_type = "Distance")
    #     if conf is None:
    #         print("ik求解失败")
    #     else:   # ik求解成功
    #         rbt_s.fk("arm", conf)
    #         # 碰撞检测
    #         if not rbt_s.is_collided(obstacle_list):    # 没碰撞
    #             rbt_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(base)  # 将机器人模型添加到场景
    #             conf_list.append(conf)
    #         else:   # 碰撞
    #             rbt_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(base)  # 将机器人模型添加到场景

        # print(jaw_width)
        # rbt_s.jaw_to(jaw_width)
        # test_conf = rbt_s.ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat)
        # rbt_s.fk('arm', test_conf)
        # rbt_s.gen_meshmodel().attach_to(base)

    # U625初始物体
    obj_pos = CONFIG_U625['grasp']['box_pos'] + np.array([0.1, 0.14, 0.05])
    obj_rot = rm.rotmat_from_axangle([0, 0, 1], 0)  # wait_rot
    # 计算手爪无碰撞的抓取中心列表
    jaw_center_info_list = object_jaw.U625_to_jaw(obj_pos, obj_rot)
    # base.run()
    conf_list = []
    seed = CONFIG_U625['grasp']['box_center_conf']
    for jaw_center_info in jaw_center_info_list:  # 不碰撞的手爪列表
        jaw_center_pos, jaw_center_rotmat = jaw_center_info
        conf = rbt_s.tracik(tgt_pos=jaw_center_pos,
                            tgt_rotmat=jaw_center_rotmat,
                            seed_jnt_values=seed,
                            solver_type="Distance")
        if conf is None:
            print("ik求解失败")
        else:  # ik求解成功
            rbt_s.fk("arm", conf)
            # 碰撞检测
            if not rbt_s.is_collided(obstacle_list):  # 没碰撞
                rbt_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(base)  # 将机器人模型添加到场景
                conf_list.append(conf)
            else:  # 碰撞
                print("ik结果碰撞")
                rbt_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(base)  # 将机器人模型添加到场景

    base.run()