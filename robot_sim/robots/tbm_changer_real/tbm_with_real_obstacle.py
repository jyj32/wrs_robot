import os
import math
import numpy as np
import robot_sim.robots.robot_interface as ri
import robot_sim._kinematics.jlchain as jl
import modeling.model_collection as mc
import robot_sim.manipulators.tbm_arm_real.tbm_arm_with_gripper as tbma
import robot_sim.end_effectors.gripper.tbm_gripper_real.tbm_gripper_with_fingers as tbmg

class TBM_Changer_Real(ri.RobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="tbm", enable_cc=True):
        super().__init__(pos=pos,rotmat=rotmat,name=name)
        this_dir,this_filename = os.path.split(__file__)

        # base_plate
        self.base_plate = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_plate',
                                     cdprimitive_type = 'box',
                                     cdmesh_type = 'aabb'
                                     )

        self.base_plate.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.base_plate.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "tbm_full_nodoor.stl")
        self.base_plate.lnks[0]['scale'] = [0.001,0.001,0.001]
        # 对心位置
        #self.base_plate.lnks[0]['loc_pos'] = np.array([2.835, -0.887, -3.1])
        self.base_plate.lnks[0]['loc_pos'] = np.array([2.835, -0.92, -3.1])
        self.base_plate.lnks[0]['loc_rotmat'] = np.dot(np.array([[-1, 0, 0],[0,-1,0],[0,0,1]]),np.array([[1,0,0],[0,0,-1],[0,1,0]]))
        self.base_plate.lnks[0]['rgba'] = [.7, .7, .7, 1]
        self.base_plate.reinitialize()

        # arm
        arm_homeconf = np.zeros(7)
        # arm_homeconf[1] = -math.pi / 6
        # arm_homeconf[2] = math.pi / 2
        # arm_homeconf[4] = math.pi / 6
        self.arm = tbma.TBMArm(pos=pos,
                               rotmat=rotmat,
                               homeconf=arm_homeconf,
                               name='arm',
                               enable_cc=False)

        #gripper
        self.hnd = tbmg.TBMGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                       rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                       name='hnd_s', enable_cc=False)

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self .enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        # self.cc.add_cdlnks(self.base_plate, [0])
        self.cc.add_cdlnks(self.arm, [ 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft_fgr, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt_fgr, [1])
        activelist = [
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft_fgr.lnks[0],
                      self.hnd.lft_fgr.lnks[1],
                      self.hnd.rgt_fgr.lnks[1]
                     ]
        self.cc.set_active_cdlnks(activelist)

        fromlist = [
                    self.arm.lnks[1]
                    ]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_fgr.lnks[0],
                    self.hnd.lft_fgr.lnks[1],
                    self.hnd.rgt_fgr.lnks[1]
                    ]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [
                    self.arm.lnks[2],
                    ]
        intolist = [
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_fgr.lnks[0],
                    self.hnd.lft_fgr.lnks[1],
                    self.hnd.rgt_fgr.lnks[1]
        ]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3],
                    ]
        intolist = [
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_fgr.lnks[0],
                    self.hnd.lft_fgr.lnks[1],
                    self.hnd.rgt_fgr.lnks[1]
                    ]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [
                    self.arm.lnks[4],
                    ]
        intolist = [
                    self.arm.lnks[6],
                    self.hnd.lft_fgr.lnks[0],
                    self.hnd.lft_fgr.lnks[1],
                    self.hnd.rgt_fgr.lnks[1]
        ]
        self.cc.set_cdpair(fromlist, intolist)

        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_plate.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_plate.jnts[-1]['gl_posq'], rotmat=self.base_plate.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat


    def fk(self, component_name='arm', jnt_values=np.zeros(7)):
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
            #七自由度需要更改关节的大小
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move the arm!")
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

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)
        #

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
        intolist = [self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat




    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='tbm_changer_real_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_plate.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
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
                      name='tbm_changer_real_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.base_plate.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
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


if __name__=='__main__':

    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm
    import motion.probabilistic.rrt_connect as rrtc
    import csv
    import motion.probabilistic.rrt_star_connect as rrtsc
    from direct.task.TaskManagerGlobal import taskMgr
    import modeling.collision_model as cm

    base = wd.World(cam_pos=[10, 0, 2], lookat_pos=[3, 0, 0])
    gm.gen_frame().attach_to(base)

    robot_s = TBM_Changer_Real(enable_cc=True)
    robot_s.gen_stickmodel(toggle_tcpcs=True).attach_to(base)
    robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_meshmodel.attach_to(base)
    robot_s.show_cdprimit()
    # 设置障碍物实现rrt
    # object_virtual
    # object
    # object_list = []
    # object_box = cm.gen_box(extent=[4, .025, .025])
    # pos = [2.835, -0.88, 0.3]
    # object_box.set_pos(np.array(pos))
    # object_box.set_rgba([.3, .6, .5, 0.8])
    # # object_box.attach_to(base)
    # # 一共多少个立方体组成圆环
    # n = 100
    # count = 0
    # # 循环创建并复制十个 box 对象，并设置它们的位置
    # for i in range(n):
    #     # 计算每个 box 的旋转角度（围绕 x 轴）
    #     angle = i * (2 * math.pi / n)  # 360度分成10份
    #
    #     # 复制一个新的 box
    #     new_box = object_box.copy()
    #
    #     # 计算新 box 的位置（围绕原点）
    #     new_x = 2  # x 轴不变
    #     new_y = 0.45 * math.cos(angle)  # 沿 y 轴移动
    #     new_z = 0.45 * math.sin(angle)+pos[2]  # 沿 z 轴移动
    #     if new_y>0.38 or new_y<-0.38 or new_z>0.1 :
    #
    #         # new_x = 4  # x 轴不变
    #         # new_y = 0.6 * math.cos(angle)  # 沿 y 轴移动
    #         # new_z = 0.6 * math.sin(angle) + pos[2]  # 沿 z 轴移动
    #         # 设置新 box 的位置
    #         new_box.set_pos(np.array([new_x, new_y, new_z]))
    #         # 将新 box 对象添加到场景中
    #         new_box.attach_to(base)
    #         new_box.show_cdprimit()
    #         object_list.append(new_box)
    #         count = count+1
    #         object_list = object_list[0:count + 1]


    base.run()


