import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur7e.ur7e as rbt
# import robot_sim.end_effectors.gripper.dh50longfinger.dh50longfinger as hnd
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
from grasp_large_bolt.config import CONFIG
from trac_ik import TracIK
from typing import Literal
from scipy.spatial.transform import Rotation as Rot

class UR7E(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur7e", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)

        self.base_stand2 = jl.JLChain(pos=pos+np.array([0.6125,-0.5875,0]), # 机器人底座原点位置
                                      rotmat=rotmat,
                                      homeconf=np.zeros(0),
                                      name='base_stand2')
        self.base_stand2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "connnect.stl"),
            cdprimit_type="box", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand2.reinitialize()

        self.waican = True  # 是否用于外参标定
        if self.waican:
            # 添加障碍物(桌子)
            self.get_obstacle_list(base,False)
            # 用于标定外参
            # cube1 = cm.CollisionModel("meshes/3cm.stl",
            #                           cdprimit_type="box", expand_radius=.001)
            # cube2 = cm.CollisionModel("meshes/3cm.stl",
            #                           cdprimit_type="box", expand_radius=.001)
            # cube3 = cm.CollisionModel("meshes/3cm.stl",
            #                           cdprimit_type="box", expand_radius=.001)
            # cube4 = cm.CollisionModel("meshes/3cm.stl",
            #                           cdprimit_type="box", expand_radius=.001)
            # cube1_pos = pos + np.array([0.6125,0,0])
            # cube2_pos = pos + np.array([0.6125+0.075, 0, 0])
            # cube3_pos = pos + np.array([0.6125+0.075, 0.1, 0])
            # cube4_pos = pos + np.array([0.6125, 0.1, 0])
            # cube1.set_pos(cube1_pos)
            # cube2.set_pos(cube2_pos)
            # cube3.set_pos(cube3_pos)
            # cube4.set_pos(cube4_pos)
            # cube1.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
            # cube2.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
            # cube3.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
            # cube4.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
            # cube1.set_rgba([0, 1, 0, 1])
            # cube2.set_rgba([0, 1, 0, 1])
            # cube3.set_rgba([0, 1, 0, 1])
            # cube4.set_rgba([0, 1, 0, 1])
            # cube1.attach_to(base)
            # cube2.attach_to(base)
            # cube3.attach_to(base)
            # cube4.attach_to(base)
            # # 展示物体坐标系
            # for cube in [cube1, cube2, cube3, cube4]:
            #     cube_pos = cube.get_pos()
            #     cube_rot = cube.get_rotmat()
            #     cube_frame = gm.gen_frame(length=0.05, pos=cube_pos, rotmat=cube_rot)
            #     cube_frame.attach_to(base)
        # arm
        arm_homeconf = np.array([3*math.pi/4,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2, 0])
        self.arm_homeconf = arm_homeconf
        self.arm_pos = pos+np.array([0.6125,-0.5875,0.015]) # 机器人位置
        self.arm_rotmat = self.base_stand2.jnts[-1]['gl_rotmatq']

        self.arm = rbt.UR7E(pos=self.arm_pos,   # 机器人位置
                             rotmat=self.arm_rotmat,    # 机器人坐标系在世界坐标系的旋转矩阵
                             homeconf=arm_homeconf,
                             name='arm', enable_cc=True)
        # gripper
        self.hnd = hnd.Dh50(pos=self.arm.jnts[-1]['gl_posq'],
                            rotmat= self.arm.jnts[-1]['gl_rotmatq'] , # 手爪安装的旋转角度
                            name='hnd', enable_cc=False)
        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd
        self.iksolver_cache = {}

    def get_obstacle_list(self, base0, show_collision = True):  # 加入碰撞体

        # 读当前文件目录
        DIR_ROOT, _ = os.path.split(__file__)
        table = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "wholetable.stl"),
            cdprimit_type="box", expand_radius=0.01
        )
        box1 = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "box1.stl"),
            cdprimit_type="box", expand_radius=0.003
        )
        box1.set_pos(np.array([0.32, 0.22, 0]))
        box2 = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "box2.stl"),
            cdprimit_type="box", expand_radius=0.003
        )
        box2.set_pos(np.array([0.32, 0.22, 0]))
        box3 = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "box3.stl"),
            cdprimit_type="box", expand_radius=0.003
        )
        box3.set_pos(np.array([0.32, 0.22, 0]))
        box4 = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "box4.stl"),
            cdprimit_type="box", expand_radius=0.003
        )
        box4.set_pos(np.array([0.32, 0.22, 0]))
        Mecheye = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "Mecheye.STL"),
            cdprimit_type="box", expand_radius=0.05  # 碰撞体扩大
        )
        Mecheye.set_pos(np.array([0.53, -0.07, 1.02]))

        M16table = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "M16table.STL"),
            cdprimit_type="box", expand_radius=0.001  # 碰撞体扩大
        )
        M16table.set_pos(np.array([0.6125+0.025*3-0.0625,-0.5875+0.45,0]))
        M16table.set_rotmat(rm.rotmat_from_euler(0, 0, np.pi))

        M16hole = cm.CollisionModel(
            os.path.join(DIR_ROOT, "meshes", "M16hole.STL"),
            cdprimit_type="box", expand_radius=0.001  # 碰撞体扩大
        )
        M16hole.set_pos(np.array([0.6125, -0.5875+0.45+0.025*7, 0]))
        M16hole.set_rotmat(rm.rotmat_from_euler(0, 0, np.pi/2))

        obstacle_list = [table,
                         # box1,
                         # box2,
                         # box3,
                         # box4,
                         Mecheye,
                         M16table,
                         M16hole,
                         ]
        for obstacle in obstacle_list:
            obstacle.attach_to(base0)
        if show_collision:  # 展示碰撞体
            for obstacle in obstacle_list:
                obstacle.show_cdprimit()  # 展示碰撞体
        return obstacle_list

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.1, 0.0, 0.14 - 0.82),
                                              x=.35 + radius, y=.3 + radius, z=.14 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3),
                                              x=.112 + radius, y=.112 + radius, z=.3 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand2, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft,[0,1])
        self.cc.add_cdlnks(self.hnd.rgt,[1])
        activelist = [self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1],
                      ]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.arm.lnks[0],
                    self.arm.lnks[1]]   # 中间必须少一个，不然一直检测到碰撞
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1],]
        self.cc.set_cdpair(fromlist, intolist)  # 设置碰撞检测对
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold("arm",objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.arm.fix_to(pos=self.base_stand2.jnts[-1]['gl_posq'], rotmat=self.base_stand2.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])

        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    # def jaw_center_pos(self):
    #     return self.machine.jaw_center_pos

    # def jaw_center_rot(self):
    #     return self.machine.jaw_center_rot

    def get_tgt_pose_in_rbt(self, tgt_pos, tgt_rotmat):
        '''
        将目标位姿（通常是指手爪中心需要到达的位置和旋转矩阵）从世界坐标系转换到机器人基坐标系下的末端执行器位姿。
        输入：
            tgt_pos:世界坐标系下目标位置
            tgt_rotmat:世界坐标系下目标旋转矩阵
        输出：
            new_tgt_pos:机器人坐标系下目标位置
            new_tgt_rot:机器人坐标系下目标旋转矩阵
        '''
        arm_pos = self.arm.pos  # 机器人相对于世界坐标系的位置
        arm_rot = self.arm.rotmat   # 机器人相对于世界坐标系的旋转矩阵
        wd_to_rbt = rm.homomat_from_posrot(arm_pos, arm_rot)    # 机器人坐标系到世界坐标系的齐次变换矩阵，名字应改为rbt_to_wd
        hand_pos = self.hnd.jaw_center_pos  # 手爪中心相对于手爪坐标系的位置
        hand_rot = self.hnd.jaw_center_rotmat   # 手爪中心相对于手爪坐标系的旋转矩阵
        # end_to_hand = rm.homomat_from_posrot(hand_pos.dot(hand_rot), hand_rot)  # 手爪坐标系到手爪中心的齐次变换矩阵，有问题？
        end_to_hand = rm.homomat_from_posrot(hand_pos, hand_rot)    # 手爪中心到手爪坐标系（机器人末端）的齐次变换矩阵
        hand_to_end = np.linalg.inv(end_to_hand)    # 手爪坐标系（机器人末端）到手爪中心的齐次变换矩阵
        tgt_homomat_wd = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)    # 目标（手爪中心）到世界坐标系的齐次变换矩阵
        end_homomat_wd = tgt_homomat_wd.dot(hand_to_end)    # 手爪坐标系（机器人末端）到世界坐标系的齐次变换矩阵
        end_homomat_rbt = np.linalg.inv(wd_to_rbt).dot(end_homomat_wd)  # 手爪坐标系（机器人末端）到机器人坐标系的齐次变换矩阵
        new_tgt_rot = end_homomat_rbt[:3, :3]   # 手爪坐标系（机器人末端）相对于机器人坐标系的旋转矩阵
        new_tgt_pos = end_homomat_rbt[:3, 3]    # 手爪坐标系（机器人末端）相对于机器人坐标系的位置
        return new_tgt_pos, new_tgt_rot

    def tracik(self,
               urdf_path: str = os.path.join(os.path.dirname(__file__), "urdf/ur7e.urdf"),
               base_link_name: str = 'base_link',
               tip_link_name: str = 'wrist_3_link',
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None,
               solver_type: Literal['Speed', 'Distance', 'Manip1', 'Manip2'] = "Distance"):
        # Distance：选取最接近seed的关节点
        new_tgt_pos, new_tgt_rot = self.get_tgt_pose_in_rbt(tgt_pos, tgt_rotmat)
        key = (urdf_path, base_link_name, tip_link_name, solver_type)
        if key not in self.iksolver_cache:
            self.iksolver_cache[key] = TracIK(base_link_name=base_link_name,
                                              tip_link_name=tip_link_name,
                                              urdf_path=urdf_path,
                                              solver_type=solver_type)
        iksolver = self.iksolver_cache[key]
        seed_jnt_values = seed_jnt_values if seed_jnt_values is not None else np.zeros(6)
        return iksolver.ik(new_tgt_pos, new_tgt_rot, seed_jnt_values)

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    # def get_jnt_init(self, component_name):
    #     if component_name in self.manipulator_dict:
    #         return self.arm.init_jnts
    #     else:
    #         raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self,hand_name, jawwidth):
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        添加物体的碰撞体积到手爪上
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def grasp(self, obj_cmodel):
        try:
            self.cc.add_cdlnks(obj_cmodel, [0])  # 假设物体只有一个链接
        except ValueError as e:
            if "already added" not in str(e):
                raise e

        # 更新激活列表，包含被抓取的物体
        current_activelist = self.cc.get_active_cdlnks()  # 获取当前激活列表
        current_activelist.append(obj_cmodel.lnks[0])  # 添加物体链接
        self.cc.set_active_cdlnks(current_activelist)  # 设置新的激活列表

        # 存储物体信息
        self.oih_infos.append({
            'obj_cmodel': obj_cmodel,
            'gl_pos': obj_cmodel.get_pos(),
            'gl_rotmat': obj_cmodel.get_rotmat()
        })


    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            # print(jawwidth)
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                # self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    # def get_jaw_width(self,hnd_name):
    #     return self.hnd.get_jaw_width()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='UR7E_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand2.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=False,
                                        toggle_jntscs=toggle_jntscs,
                                        toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      is_machine=None,
                      is_robot=True,
                      name='UR7E_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand2.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                           tcp_loc_pos=tcp_loc_pos,
                                           tcp_loc_rotmat=tcp_loc_rotmat,
                                           toggle_tcpcs=False,
                                           toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)

        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

    def get_real_tcp_pose(self, tgt_pos, tgt_rotmat):
        '''
        根据手爪中心在仿真中的位置和旋转矩阵计算出机器人的实际tcp_pose
        输入:
            tgt_pos:仿真中的目标（手爪中心）位置
            tgt_rotmat:仿真中的目标（手爪中心）旋转矩阵
        输出:
            real_tcp_pose:实际的TCP_pose
        '''
        # 仿真中齐次变换矩阵
        T_base_world = rm.homomat_from_posrot(self.arm_pos, self.arm_rotmat)  # 基座→世界
        T_world_base = np.linalg.inv(T_base_world)  # 世界→基座
        # 目标（手爪中心）在世界坐标系中的齐次矩阵
        T_target_world = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        # 目标（手爪中心）在机器人基坐标系中的齐次矩阵
        T_target_base = T_world_base.dot(T_target_world)
        # 提取位置（手爪中心在机器人坐标系下的坐标）和旋转矩阵
        # pos_in_base = T_target_base[:3, 3]
        # rot_in_base = T_target_base[:3, :3]
        # print(f"pos_in_base:{pos_in_base}")
        # print(f"rot_in_base:{rot_in_base}")
        # 求手爪坐标系（机器人末端法兰盘）在机器人坐标系下的位置和旋转矩阵
        # 工具定义：手爪中心相对于法兰盘的位置和旋转（需从机器人模型中获取）
        tool_pos = self.hnd.jaw_center_pos  # [0, 0, 0.16882]
        tool_rot = self.hnd.jaw_center_rotmat  # 通常为单位矩阵
        # 构造手爪中心 → 法兰盘的齐次矩阵
        T_jaw_center_to_flange = rm.homomat_from_posrot(tool_pos, tool_rot)
        # 计算法兰盘在机器人基坐标系下的齐次矩阵
        T_flange_base = T_jaw_center_to_flange @ T_target_base
        # 提取位置和旋转矩阵
        # flange_pos_in_base = T_flange_base[:3, 3]
        # flange_rot_in_base = T_flange_base[:3, :3]
        # print(f"仿真中法兰盘位置（在机器人基坐标系下）: {flange_pos_in_base}")
        # print(f"仿真中法兰盘旋转矩阵（在机器人基坐标系下）:\n{flange_rot_in_base}")

        # rotvec_base = Rot.from_matrix(rot_in_base).as_rotvec()  # 把旋转矩阵转化为轴角
        # print("在仿真环境基座坐标系下TCP轴角 (rx, ry, rz):", rotvec_base)
        # 计算结果与实际不一样甚至相差很大的原因：机器人的仿真的机器人坐标系和实际的机器人坐标系不一样，采用仿真中的机器人位置和旋转矩阵，仿真中的目标的位置和旋转矩阵，计算出仿真的目标位姿，与实际的目标位姿有很大误差。
        # 所以需要仿真机器人坐标系到真实机器人坐标系的转换
        # # 6.定义仿真到真实机器人坐标系的变换矩阵,可根据仿真中轴角和位置和实际机器人的轴角和位置计算出来
        # 实际的（轴角）旋转向量
        # r_real = np.array([-2.219160348291919, -2.2199962726445013, 0.0010388431351335242])
        # # 实际的旋转矩阵（在机器人坐标系下）
        # R_real = Rot.from_rotvec(r_real).as_matrix()
        # # 仿真到实际的变换矩阵
        # R_sim_to_real = R_real @ rot_in_base.T
        # print("R_sim_to_real:", R_sim_to_real)
        R_sim_to_real = [[-9.99997212e-01 , 3.74987480e-04  ,2.33129348e-03],
                         [-3.78239181e-04 ,-9.99998956e-01 ,-1.39452108e-03],
                         [ 2.33076811e-03 ,-1.39539898e-03 , 9.99996310e-01]]  # 固定不变
        # # 计算仿真到实际的平移变换矩阵
        # # 实际的tcp在机器人坐标系的位置
        # p_sim = flange_pos_in_base  # 仿真基系下该点的位置
        # p_real = np.array([-0.3882219431355504, -0.43759623810862014, 0.3681100180615627])  # 真实基系下该点的位置
        # # 计算平移向量
        # t_sim_to_real = p_real - R_sim_to_real @ p_sim
        # print("t_sim_to_real:", t_sim_to_real)
        t_sim_to_real = [-0.001745  ,  0.00056306 ,-0.00018131] # 固定不变
        T_sim_to_real = rm.homomat_from_posrot(t_sim_to_real, R_sim_to_real)    # 仿真到实际的齐次变换矩阵
        # 转换为真实机器人坐标系下TCP的位姿
        tcp_homomat_rbt = T_sim_to_real @ T_flange_base
        tcp_pos_real = tcp_homomat_rbt[:3, 3]
        tcp_rot_real = tcp_homomat_rbt[:3,:3]
        rotvec_real = Rot.from_matrix(tcp_rot_real).as_rotvec() # 把旋转矩阵转化为轴角
        print("在真实机器人基座下TCP位置:", tcp_pos_real)
        print("TCP轴角 (rx, ry, rz):", rotvec_real)
        real_tcp_pose = np.concatenate([tcp_pos_real, rotvec_real])  # shape (6,)
        return real_tcp_pose

if __name__ == '__main__':

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])   # 仿真环境
    gm.gen_frame().attach_to(base)  # 世界坐标系（0,0,0）
    robot_s = UR7E(enable_cc=True)  # 创建UR7E机器人实例，启用碰撞检测,机器人原点位置默认在[0.6125,-0.5875,0]
    robot_s.gen_meshmodel().attach_to(base)
    robot_s.show_cdprimit()  # 展示碰撞体
    # 绘制基坐标系（长度设为0.2米，可自行调整）
    gm.gen_frame(length=0.2, pos=robot_s.arm_pos, rotmat=robot_s.arm_rotmat).attach_to(base)

    # 添加障碍物
    # obstacle_list = robot_s.get_obstacle_list(base,True)
    #
    # base.run()
    #
    # xxxx = robot_s.ik('arm', pos, rot)
    # print(xxxx)
    # xxxx = np.array([2.3590309619903564, -1.571965833703512, 1.57050067583193, -1.569312409763672, -1.57078725496401, 0.0005946227465756238])
    # robot_s.fk(component_name='arm', jnt_values=xxxx)
    # robot_s.gen_meshmodel().attach_to(base)
    # base.run()
    # robot_s.environment_object(objct=True)  # 添加环境物体（objct=True表示添加桌子）
    # # robot_s.fk(component_name='arm',jnt_values=np.zeros(6))
    # robot_s.gen_meshmodel().attach_to(base)    # 生成机器人初始状态的网格模型并添加到仿真世界
    # pos = np.array([1, -0.2, 0.4])  # 目标位置
    # rot = rm.rotmat_from_axangle([0, 1, 0], math.pi)    # 目标姿态
    # xxxx = robot_s.ik('arm', pos, rot)  # ik求解的姿态
    # robot_s.fk(component_name='arm', jnt_values=xxxx)   # 正运动学求解
    # robot_s.gen_meshmodel().attach_to(base)   # 生成机器人初始状态的网格模型并添加到仿真世界
    # # base.run()
    # #robot_s.hnd.open() # 手爪打开
    # #robot_s.gen_meshmodel().attach_to(base)
    # base.run()

    # # 将位置和旋转矩阵转化为实际的TCP位姿
    # tgt_pos = np.array([0.75,0,0.255])   # 仿真中手爪中心位置
    # tgt_rotmat = [[ 0.0000000e+00, -1.0000000e+00,  1.2246468e-16],
    #                  [-1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
    #                  [ 0.0000000e+00, -1.2246468e-16 ,-1.0000000e+00]]  # 仿真中手爪中心姿态
    # tcp_pose = robot_s.get_real_tcp_pose(tgt_pos,tgt_rotmat)
    # print(list(tcp_pose))

    # R_sim_to_real = [[-9.99996829e-01 ,-5.02650872e-05 , 2.51778278e-03],
    #                      [ 4.75902745e-05 ,-9.99999435e-01 ,-1.06241698e-03],
    #                      [ 2.51783476e-03 ,-1.06229379e-03 , 9.99996266e-01]]
    # rotation = Rot.from_matrix(R_sim_to_real)
    # angle = rotation.as_euler('ZYX')
    # print(angle)
    # base.run()
    rot = np.dot(
        rm.rotmat_from_axangle([0, 1, 0], math.pi),
        rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    )
    print(rot)