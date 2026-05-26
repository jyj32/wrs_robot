import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur7e.ur7e as rbt
import robot_sim.end_effectors.gripper.dh50longfinger.dh50longfinger as longhnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import copy
import modeling.geometric_model as gm
import robot_sim.manipulators.machinetool.machinetool_gripper as machine
import basis.robot_math as rm
from trac_ik import TracIK
from typing import Literal
from scipy.spatial.transform import Rotation as Rot

class UR7E(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur7e", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)


        # arm
        self.arm_pos = pos
        self.arm_rotmat = rotmat
        arm_homeconf = np.array([0,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,-math.pi/2])
        # arm_homeconf = np.zeros(6)
        self.arm = rbt.UR7E(pos=pos,
                             rotmat=rotmat,
                             homeconf=arm_homeconf,
                             name='arm', enable_cc=True)
        # gripper
        self.gripper_rot = np.dot(self.arm.jnts[-1]['gl_rotmatq'],
                                  rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2*0))
        self.hnd = longhnd.Dh50(pos=self.arm.jnts[-1]['gl_posq'],
                            rotmat=self.gripper_rot,
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

        waican = False

        if waican: # 相机标定

            self.get_obstacle_list(base, False)    # 加入障碍物
            # 加入标定方块
            cube1 = cm.CollisionModel("object/3cm.stl",
                cdprimit_type="box", expand_radius=.001)
            cube2 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            cube3 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            cube4 = cm.CollisionModel("object/456.stl",
                                      cdprimit_type="box", expand_radius=.001)
            box2_pos = np.array([0.232, 0.32, 0.74])

            # U1箱子
            cube1_pos = box2_pos + np.array([0.41/ 2-0.015,  0.61 / 2-0.015, 0])
            cube2_pos = box2_pos + np.array([0.41 / 2-0.015, -0.61 / 2+0.015, 0])
            cube3_pos = box2_pos + np.array([-0.41 / 2+0.015, 0.61 / 2-0.015, 0])
            cube4_pos = box2_pos + np.array([-0.41 / 2, -0.61 / 2, 0])
            cube1.set_pos(cube1_pos)
            cube2.set_pos(cube2_pos)
            cube3.set_pos(cube3_pos)
            cube4.set_pos(cube4_pos)
            cube1.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube2.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube3.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube4.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
            cube1.set_rgba([1, 1, 0, 1])
            cube2.set_rgba([1, 1, 0, 1])
            cube3.set_rgba([1, 1, 0, 1])
            cube4.set_rgba([1, 1, 0, 1])
            cube1.attach_to(base)
            cube2.attach_to(base)
            cube3.attach_to(base)
            cube4.attach_to(base)


            cube5 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            cube6 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            cube7 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            cube8 = cm.CollisionModel("object/3cm.stl",
                                      cdprimit_type="box", expand_radius=.001)
            box1_pos = np.array([0.807, -0.245, 0.84])

            # U625箱子
            cube5_pos = box1_pos + np.array([ 0.61 / 2 - 0.015,0.41 / 2 - 0.015, 0])
            cube6_pos = box1_pos + np.array([ -0.61 / 2 + 0.015, 0.41 / 2 - 0.015,0])
            cube7_pos = box1_pos + np.array([ 0.61 / 2 - 0.015,-0.41 / 2 + 0.015, 0])
            cube8_pos = box1_pos + np.array([-0.61 / 2+ 0.015, -0.41 / 2+ 0.015, 0])
            cube5.set_pos(cube5_pos)
            cube6.set_pos(cube6_pos)
            cube7.set_pos(cube7_pos)
            cube8.set_pos(cube8_pos)
            cube5.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube6.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube7.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube8.set_rotmat(rm.rotmat_from_euler(math.pi / 2, 0, math.pi / 1))
            cube5.set_rgba([1, 1, 0, 1])
            cube6.set_rgba([1, 1, 0, 1])
            cube7.set_rgba([1, 1, 0, 1])
            cube8.set_rgba([1, 1, 0, 1])
            cube5.attach_to(base)
            cube6.attach_to(base)
            cube7.attach_to(base)
            cube8.attach_to(base)

            # 展示物体坐标系
            for cube in [cube1, cube2, cube3, cube4,cube5,cube6,cube7,cube8]:
                pos = cube.get_pos()
                rot = cube.get_rotmat()
                frame = gm.gen_frame(length=0.05, pos=pos, rotmat=rot)
                frame.attach_to(base)

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

    def get_obstacle_list(self,base0, show_collision):

        # 原有机器，大的桌子
        machine = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/allmachine2.stl",
            cdprimit_type="box", expand_radius=.001)
        collections1 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/pengzhuang1.stl",
            cdprimit_type="box", expand_radius=.001)
        collections1_pos = np.array([-0.13287+0.09, -0.36, 0.753+0.05])
        collections1.set_rgba([0.5, 0.5, 0.5, 0.2])
        collections1.set_pos(collections1_pos)

        collections2 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/pengzhuang2.stl",
            cdprimit_type="box", expand_radius=.001)
        collections2_pos = np.array([-0.095, 0.202, 0.683+0.05])
        collections2.set_rgba([0.5, 0.5, 0.5, 0.2])
        collections2.set_pos(collections2_pos)
        # D435相机下面
        collections3 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/cdprimit1.stl",
            cdprimit_type="box", expand_radius=.001)
        collections3_pos = np.array([0.155, -0.905+0.06, 0])
        collections3.set_pos(collections3_pos)
        collections3.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        # 梅卡曼德相机下面
        collections4 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/cdprimit2.stl",
            cdprimit_type="box", expand_radius=.001)
        collections4_pos = np.array([-0.14, 0.825, 0.5])
        collections4.set_pos(collections4_pos)
        collections4.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1 * 0))
        # 梅卡曼德相机
        collections5 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/cdprimit3.stl",
            cdprimit_type="box", expand_radius=.001)
        collections5_pos = np.array([-0.075, 0.425, 1.55])
        collections5.set_pos(collections5_pos)
        collections5.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1 * 0))
        # D435相机下面支架
        collections6 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/cdprimit4.stl",
            cdprimit_type="box", expand_radius=.001)
        collections6_pos = np.array([0.73, -0.525, 1.4])
        collections6.set_pos(collections6_pos)
        collections6.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1 * 0))
        # D435相机
        collections7 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/cdprimit5.stl",
            cdprimit_type="box", expand_radius=.001)
        collections7_pos = np.array([0.73, -0.325, 1.75])
        collections7.set_pos(collections7_pos)
        collections7.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1 * 0))
        # U625箱子
        box1 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148.stl",
            cdprimit_type="box", expand_radius=.001)
        box1_pos = np.array([0.807, -0.245, 0.84])
        box1.set_pos(box1_pos)
        box1.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))

        box1_collections1 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_1.stl",
            cdprimit_type="box", expand_radius=.001)
        box1_collections1.set_pos(box1_pos+np.array([0,-0.017,0]))
        box1_collections1.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))
        box1_collections2 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_2.stl",
            cdprimit_type="box", expand_radius=.001)
        box1_collections2.set_pos(box1_pos+np.array([0,-0.01,0]))
        box1_collections2.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))
        box1_collections3 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_3.stl",
            cdprimit_type="box", expand_radius=.001)
        box1_collections3.set_pos(box1_pos+np.array([-0.013,0,0+0.01]))
        box1_collections3.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))
        box1_collections4 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_4.stl",
            cdprimit_type="box", expand_radius=.001)
        box1_collections4.set_pos(box1_pos+np.array([0.013,0,0]))
        box1_collections4.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))
        box1_collections5 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_5.stl",
            cdprimit_type="box", expand_radius=.001)    # 底部
        box1_collections5.set_pos(box1_pos+np.array([0,0,-0.01]))
        box1_collections5.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 2 * 1))
        # U1箱子
        box2 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_pos = np.array([0.232, 0.32, 0.74])
        box2.set_pos(box2_pos)
        box2.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))

        box2_collections1 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_1.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_collections1.set_pos(box2_pos+np.array([0.007,0,0+0.03]))
        box2_collections1.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        box2_collections2 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_2.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_collections2.set_pos(box2_pos+np.array([-0.01,0,0+0.03]))
        box2_collections2.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        box2_collections3 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_3.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_collections3.set_pos(box2_pos+np.array([0,-0.017,0+0.03]))
        box2_collections3.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        box2_collections4 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_4.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_collections4.set_pos(box2_pos+np.array([0,0.005,0+0.03]))
        box2_collections4.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        box2_collections5 = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/600400148_5.stl",
            cdprimit_type="box", expand_radius=.001)
        box2_collections5.set_pos(box2_pos+np.array([0,0,-0.005]))
        box2_collections5.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        # 吸盘
        xipan = cm.CollisionModel(
            "E:/py_project/wrsrobot/wrs_shu/robot_sim/robots/ur7e/object/xipan.stl",
            cdprimit_type="box", expand_radius=.001)
        xipan_pos = np.array([0.22, -0.13, 0.77])
        xipan.set_pos(xipan_pos)
        xipan.set_rotmat(rm.rotmat_from_euler(0, 0, math.pi / 1))
        # 所有物体模型
        all_machine = [machine,
                       collections1,
                       collections2,
                       collections3,
                       collections4,
                       collections5,
                       collections6,
                       collections7,
                       box1_collections1, box1_collections2, box1_collections3,box1_collections4,
                       box1_collections5,
                       box2_collections1, box2_collections2, box2_collections3,box2_collections4,
                       box2_collections5,
                       xipan,
                       ]
        for obstacle in all_machine:
            obstacle.attach_to(base0)
        # 可能的碰撞体,不能加入镂空的物体
        obstacle_list = [collections1, collections2, collections3, collections4, collections5, collections6, collections7,
                       box1_collections1, box1_collections2, box1_collections3,box1_collections4,
                         box1_collections5,
                       box2_collections1, box2_collections2, box2_collections3,box2_collections4,
                         box2_collections5,
                        xipan,
                         ]
        if show_collision:  # 展示碰撞体
            for obstacle in obstacle_list:
                obstacle.show_cdprimit()

        return obstacle_list

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt, [1])
        activelist = [self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1]
                      ]
        self.cc.set_active_cdlnks(activelist)

        fromlist = [self.arm.lnks[0],
                    self.arm.lnks[1]]  # 中间必须少一个，不然一直检测到碰撞
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1], ]
        self.cc.set_cdpair(fromlist, intolist)  # 设置碰撞检测对

        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold("arm",objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.gripper_rot = np.dot(self.arm.jnts[-1]['gl_rotmatq'],
                                  rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2*0))
        self.arm.fix_to(pos=self.pos, rotmat=self.rotmat)
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.gripper_rot)

        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    # def jaw_center_pos(self):
    #     return self.machine.jaw_center_pos

    # def jaw_center_rot(self):
    #     return self.machine.jaw_center_rot

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
                rotmat=np.dot(self.arm.jnts[-1]['gl_rotmatq'],
                                  rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2*0)))
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

    def jaw_to(self,hand_name, jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
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
            print(jawwidth)
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def hndclose(self):
        self.hnd.jaw_to(0)

    def hndopen(self):
        self.hnd.jaw_to(0.076)

    def get_tgt_pose_in_rbt(self, tgt_pos, tgt_rotmat):
        arm_pos = self.arm.pos
        arm_rot = self.arm.rotmat
        wd_to_rbt = rm.homomat_from_posrot(arm_pos, arm_rot)
        hand_pos = self.hnd.jaw_center_pos
        hand_rot = self.hnd.jaw_center_rotmat
        end_to_hand = rm.homomat_from_posrot(hand_pos.dot(hand_rot), hand_rot)
        hand_to_end = np.linalg.inv(end_to_hand)
        tgt_homomat_wd = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        end_homomat_wd = tgt_homomat_wd.dot(hand_to_end)
        end_homomat_rbt = np.linalg.inv(wd_to_rbt).dot(end_homomat_wd)
        new_tgt_rot = end_homomat_rbt[:3, :3]
        new_tgt_pos = end_homomat_rbt[:3, 3]
        return new_tgt_pos, new_tgt_rot


    def tracik(self,
               urdf_path: str = os.path.join(os.path.dirname(__file__), "urdf/ur7e.urdf"),
               base_link_name: str = 'base_link',
               tip_link_name: str = 'wrist_3_link',
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None,
               solver_type: Literal['Speed', 'Distance', 'Manip1', 'Manip2'] = "Distance"):
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

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
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
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
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
        # 构造手爪中心 → 法兰盘（手爪坐标系）的齐次矩阵
        T_jaw_center_to_flange = rm.homomat_from_posrot(tool_pos, tool_rot)
        # 计算法兰盘（手爪坐标系）在机器人基坐标系下的齐次矩阵
        T_flange_base = T_jaw_center_to_flange @ T_target_base
        # 提取位置和旋转矩阵
        # flange_pos_in_base = T_flange_base[:3, 3]
        # flange_rot_in_base = T_flange_base[:3, :3]
        # print(f"仿真中法兰盘位置（在机器人基坐标系下）: {flange_pos_in_base}")
        # print(f"仿真中法兰盘旋转矩阵（在机器人基坐标系下）:\n{flange_rot_in_base}")

        # rotvec_base = Rot.from_matrix(rot_in_base).as_rotvec()
        # print("在仿真环境基座坐标系下TCP轴角 (rx, ry, rz):", rotvec_base)
        # 计算结果与实际不一样甚至相差很大的原因：机器人的仿真的位姿和实际位姿不一样，采用仿真中的机器人位置和旋转矩阵，仿真中的目标的位置和旋转矩阵，计算出仿真的目标位姿，与实际的目标位姿有很大误差。
        # 所以需要仿真机器人坐标系到真实机器人坐标系的转换
        # # 6.定义仿真到真实机器人坐标系的变换矩阵,可根据仿真中轴角和位置和实际机器人的轴角和位置计算出来
        # 实际的（轴角）旋转向量
        # r_real = np.array([0.00035475181486636215, -3.139188838850358, -0.0014876764903980694])
        # # 实际的旋转矩阵（在机器人坐标系下）
        # R_real = Rot.from_rotvec(r_real).as_matrix()
        # # 仿真到实际的变换矩阵
        # R_sim_to_real = R_real @ rot_in_base.T
        # print("R_sim_to_real:", R_sim_to_real)
        R_sim_to_real = [[-9.99997086e-01, 2.24875575e-04, 2.40354670e-03],
                        [-2.27153578e-04, -9.99999525e-01, -9.47536292e-04],
                        [2.40333248e-03, -9.48079506e-04, 9.99996663e-01]]  # 固定不变
        # # 计算仿真到实际的平移变换矩阵
        # 实际的tcp在机器人坐标系的位置
        # p_sim = flange_pos_in_base  # 仿真基系下该点的位置
        # p_real = np.array([-0.3005388549670675, -0.3000454418285177, 0.5093237804093638])  # 真实基系下该点的位置
        # # 计算平移向量
        # t_sim_to_real = p_real - R_sim_to_real @ p_sim
        # print("t_sim_to_real:", t_sim_to_real)
        t_sim_to_real = [-1.83016447e-03 , 5.04687161e-04  ,6.89024495e-05] # 固定不变
        T_sim_to_real = rm.homomat_from_posrot(t_sim_to_real, R_sim_to_real)    # 仿真到实际的齐次变换矩阵
        # 转换为真实机器人坐标系下TCP的位姿
        tcp_homomat_rbt = T_sim_to_real  @ T_flange_base
        tcp_pos_real = tcp_homomat_rbt[:3, 3]
        tcp_rot_real = tcp_homomat_rbt[:3,:3]
        rotvec_real = Rot.from_matrix(tcp_rot_real).as_rotvec()
        print("在真实机器人基座下TCP位置:", tcp_pos_real)
        print("TCP轴角 (rx, ry, rz):", rotvec_real)
        real_tcp_pose = np.concatenate([tcp_pos_real, rotvec_real])  # shape (6,)
        return real_tcp_pose

if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)
    robot_s = UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                     enable_cc=True)  # 仿真机器人
    # robot_s.gen_meshmodel().attach_to(base)
    # conf = np.array([-0.46189231, - 1.55124523,  1.75405697, - 1.77360785, - 1.57079618, - 3.60348501])
    # robot_s.fk(component_name='arm', jnt_values=conf)
    # # robot_s.gen_meshmodel().attach_to(base)
    #
    # robot_s.hnd.jaw_to(0)
    # 加入障碍物
    robot_s.get_obstacle_list(base,True)
    #
    # base.run()


    # # 通过世界坐标系的手爪中心位置和旋转矩阵，求得实际的机器人坐标系的tcp位姿pose
    # # 已知世界到机器人基座的变换（注意：实际构造的是 机器人基座→世界，这里取其逆）
    # arm_pos = np.array([0.7, 0.2, 0.7])
    # arm_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi) # 仿真中
    # T_base_world = rm.homomat_from_posrot(arm_pos, arm_rot)  # 基座→世界
    # T_world_base = np.linalg.inv(T_base_world)  # 世界→基座
    #
    # tgt_pos = np.array([0.265, -0.354, 1.01])   # 仿真中手爪中心位置
    # tgt_rotmat = [[ -1.0000000e+00 , 0.0000000e+00 , 1.2246468e-16],
    #              [ 0.0000000e+00 , 1.0000000e+00 , 0.0000000e+00],
    #              [-1.2246468e-16 , 0.0000000e+00 ,-1.0000000e+00]]  # 仿真中手爪中心姿态
    # # 目标（手爪中心）在世界坐标系中的齐次矩阵
    # T_target_world = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
    #
    # # 目标（手爪中心）在机器人基坐标系中的齐次矩阵
    # T_target_base = T_world_base.dot(T_target_world)
    #
    # # 提取位置（手爪中心在基系下的坐标）和旋转矩阵
    # pos_in_base = T_target_base[:3, 3]
    # rot_in_base = T_target_base[:3, :3]
    # print(f"pos_in_base:{pos_in_base}")
    # print(f"rot_in_base:{rot_in_base}")
    # from scipy.spatial.transform import Rotation as Rot
    # rotvec_base = Rot.from_matrix(rot_in_base).as_rotvec()
    # print("在仿真环境基座坐标系下TCP轴角 (rx, ry, rz):", rotvec_base)
    # # 计算结果与实际不一样甚至相差很大的原因：机器人的仿真的位姿和实际位姿不一样，采用仿真中的机器人位置和旋转矩阵，仿真中的目标的位置和旋转矩阵，计算出仿真的目标位姿，与实际的目标位姿有很大误差。
    # # 所以需要仿真机器人坐标系到真实机器人坐标系的转换
    # # # 6.定义仿真到真实机器人坐标系的变换矩阵,可根据仿真中轴角和实际机器人的轴角计算出来
    # # 实际的（轴角）旋转向量
    # # r_real = np.array([0.00035475181486636215, -3.139188838850358, -0.0014876764903980694])
    # # # 实际的旋转矩阵（在机器人坐标系下）
    # # R_real = Rot.from_rotvec(r_real).as_matrix()
    # # # 仿真到实际的变换矩阵
    # # R_AB = R_real @ rot_in_base.T
    # # print("R_AB:", R_AB)
    # R_AB = [[-9.99997086e-01 , 2.24875575e-04 , 2.40354670e-03],
    #          [-2.27153578e-04 ,-9.99999525e-01 ,-9.47536292e-04],
    #          [ 2.40333248e-03, -9.48079506e-04 , 9.99996663e-01]]
    # # 7.转换为真实机器人坐标系下的位姿
    # tcp_pos_real = R_AB @ pos_in_base + np.array([0,0,0.16882])  # 加上手爪的长度
    # tcp_rot_real = R_AB @ rot_in_base
    # rotvec_real = Rot.from_matrix(tcp_rot_real).as_rotvec()
    # print("在真实机器人基座下TCP位置:", tcp_pos_real)
    # print("TCP轴角 (rx, ry, rz):", rotvec_real)

    # tgt_pos = np.array([0.4, -0.1, 1.04])   # 仿真中手爪中心位置
    # tgt_rotmat = [[ -1.0000000e+00 , 0.0000000e+00 , 1.2246468e-16],
    #              [ 0.0000000e+00 , 1.0000000e+00 , 0.0000000e+00],
    #              [-1.2246468e-16 , 0.0000000e+00 ,-1.0000000e+00]]  # 仿真中手爪中心姿态
    # tcp_pose = robot_s.get_real_tcp_pose(tgt_pos,tgt_rotmat)
    # print(list(tcp_pose))

    # R_sim_to_real = [[-9.99997086e-01, 2.24875575e-04, 2.40354670e-03],
    #                  [-2.27153578e-04, -9.99999525e-01, -9.47536292e-04],
    #                  [2.40333248e-03, -9.48079506e-04, 9.99996663e-01]]
    # Rotation = Rot.from_matrix(R_sim_to_real)
    # euler = Rotation.as_euler('xyz')
    # print(euler)    # [-9.48082386e-04 -2.40333479e-03 -3.14136550e+00]