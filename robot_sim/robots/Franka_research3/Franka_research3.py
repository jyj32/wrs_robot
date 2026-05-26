import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))#根目录错误时
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.Franka_research3.Franka_research3 as rbt
import robot_sim.end_effectors.gripper.frank_research3.frank_research3 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.manipulators.machinetool.machinetool_gripper as machine
import basis.robot_math as rm
import motion.probabilistic.rrt_connect as rrtc
# 装配上手爪，桌子
# 控制程序
class Franka_research3(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="Franka_research3", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate,机器人组件初始化
        self.base_stand = jl.JLChain(pos=pos+np.array([-0.2, 0, 0]),    # 只改桌子位置
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand')
        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "table.STL"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)    # 设置碰撞检测参数
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]    # 颜色
        self.base_stand.reinitialize()
        # arm机械臂初始化
        arm_homeconf = np.zeros(7)  # home点
        # arm_homeconf[0] = math.pi / 2
        # arm_homeconf[1] = -math.pi * 1 / 3
        # arm_homeconf[2] = math.pi * 1 / 3
        # arm_homeconf[3] = -math.pi * 0 / 2
        # arm_homeconf[4] = -math.pi * 0 / 2
        self.arm = rbt.Franka(pos=pos,
                            rotmat=self.base_stand.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,  # 机器人初始关节点
                            name='arm', enable_cc=False)
        # gripper手爪初始化，固定于机械臂末端
        self.hnd = hnd.frank_research3(pos=self.arm.jnts[-1]['gl_posq'],
                                rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                name='hnd_s', enable_cc=False)
        # tool center point工具中心点
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

    @staticmethod
    def _base_combined_cdnp(name, radius):# 静态方法，定义基座的碰撞检测几何形状，使用 CollisionBox 定义碰撞体
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.65, 0.0, -0.45), # 桌子碰撞体位置
                                              x=0.7 + radius, y=0.75 + radius, z=0.45 + radius)   # 桌子碰撞体大小
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    def enable_cc(self):    # 碰撞检测
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        # 添加各组件的碰撞体到碰撞检测系统
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt, [1])

        activelist = [self.base_stand.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.arm.lnks[7],
                      self.hnd.lft.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1]]
        self.cc.set_active_cdlnks(activelist)
        # 设置碰撞检测对
        fromlist = [self.base_stand.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.arm.lnks[7],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos: # 抓住物体后，添加碰撞体
            objcm = oih_info['collision_model']
            self.hold(objcm = objcm)

    def fix_to(self, pos, rotmat):  # 固定机器人到指定位置和姿态
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'], rotmat=self.base_stand.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(7)):
        """
        正运动学
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
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):   # 手爪开到jawwidth宽度
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None): # 抓取物体并将其添加到碰撞检测系统
        """
        the objcm is added as a part of the robot_s to the cd checker
        输入：
        hnd_name: 手爪名
        jawwidth: 宽度
        objcm:  抓住的物体
        输出：
        rel_pos:
        rel_rotmat:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.base_stand.lnks[0],
                    self.arm.lnks[1],
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

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        释放物体并从碰撞检测系统中移除
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        # 生成机器人的杆状模型（用于简化显示），手爪上也会生成杆状模型

        stickmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
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
        # self.hnd.gen_stickmodel(toggle_tcpcs=False,
        #                         toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      is_robot=True,
                      is_machine=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        # 生成机器人的网格模型（用于真实感显示）
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            self.base_stand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
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

    # def door_to(self, door_width):
    #     self.machine.door_to(door_width)
    #
    # def chunck_to(self, chunck_width):
    #     self.machine.jaw_to(chunck_width)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])

    gm.gen_frame().attach_to(base)
    robot_s = Franka_research3(pos=np.array([0, 0, 0]), enable_cc=True) # 改整个机器人系统的位置

    robot_s.jaw_to(jawwidth=0.05)    # 手爪开合
    robot_s.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)

    tgt_pos = np.array([.6, .5, .2])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # robot_s.show_cdprimit() # 展示碰撞体
    # robot_s.gen_stickmodel().attach_to(base)
    # base.run()

    component_name = 'arm'  # 写arm和hnd都行
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)    # 逆解
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=False)
    robot_s_meshmodel.attach_to(base)
    # # robot_s.show_cdprimit() # 展示碰撞体
    # # robot_s.gen_stickmodel().attach_to(base)  # 机器人生成杆状模型
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)

    rrtc_planner = rrtc.RRTConnect(robot_s)  # 创建RRT-Connect路径规划器实例，用于后续的运动规划
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=np.array([0, 0, 0, 0, 0, 0, 0]),
                             goal_conf=jnt_values,
                             ext_dist=0.1,
                             max_time=300)
    i=0
    for jnts in path:
        i += 1
        if i%3==1:
            robot_s.fk("arm", jnts)  ## 用正运动学（fk）更新机器人姿态（根据关节角度计算手臂位置）
            robot_s.gen_meshmodel(rgba=[0, 1, 0, .4]).attach_to(base)

    base.run()
