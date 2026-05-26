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

        self.base_plate.jnts[1]['loc_pos'] = np.array([0,0,0])
        self.base_plate.lnks[0]['loc_pos'] = np.array([1, 0.5, -0.35])

        # 实现基座绕x轴180，z轴180
        test_base_rotmat_x = rm.rotmat_from_axangle([1, 0, 0], math.pi)
        test_base_rotmat_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 1 / 2)
        test_base_rotmat_z = rm.rotmat_from_axangle([0, 0, 1], math.pi)
        test_base_rotmat = np.dot(test_base_rotmat_x, test_base_rotmat_z)

        self.base_plate.lnks[0]['loc_rotmat'] = test_base_rotmat
        self.base_plate.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "testplatform.stl")
        self.base_plate.lnks[0]['scale'] = [0.001,0.001,0.001]
        self.base_plate.reinitialize()

        # arm
        arm_homeconf = np.zeros(7)
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
        self.cc.add_cdlnks(self.base_plate, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft_fgr, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt_fgr, [1])
        activelist = [
                      self.base_plate.lnks[0],
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
        fromlist = [self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_fgr.lnks[0],
                    self.hnd.lft_fgr.lnks[1],
                    self.hnd.rgt_fgr.lnks[1]]
        intolist = [self.base_plate.lnks[0]]
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
    gm.gen_frame(length=1, thickness=0.1).attach_to(base)

    robot_s = TBM_Changer_Real(enable_cc=True)
    robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_meshmodel.show_cdprimit()
    robot_meshmodel.attach_to(base)
    print(robot_s.get_gl_tcp("arm"))
    pos = robot_s.get_gl_tcp("arm")[0]
    rot = robot_s.get_gl_tcp("arm")[1]
    gm.gen_frame(pos, rotmat=rot, length=1, thickness=0.1).attach_to(base)
    # 单次ik
    #ik 由关节坐标得到笛卡尔坐标
    # tgt_pos = np.array([ 4, 0,0])
    # # [4.5     0.125 - 0.4375]
    # # jnt_values[2.0942797
    # # 0.22300167
    # # 1.15193624 - 0.69293421 - 1.22095702 - 0.63040929
    # # 1.71241901]
    # # jnt_values[1.38171554 - 0.07757681 - 0.92544062
    # # 0.24191461
    # # 1.26116993 - 0.20466445
    # # - 1.91399456]
    # tgt_rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi * 1 / 2)
    # gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat,length=0.5).attach_to(base)
    # component_name = 'arm'
    # jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    # print('jnt_values',jnt_values)
    # # fk 由笛卡尔坐标得到关节坐标并且展示新位姿
    # robot_s.fk(component_name, jnt_values=jnt_values)
    # robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    # robot_s_meshmodel.attach_to(base)
    # #robot_s.show_cdprimit()
    # base.run()
    # [3.   1. - 0.5]
    # jnt_values[1.28808952e+00
    # 1.13836823e-03 - 1.00408629e+00
    # 8.60453002e-01
    # - 3.92410327e-01 - 1.15079820e+00 - 7.76081432e-01]
    # [3.          1. - 0.42857143]
    # jnt_values[1.22784139 - 0.00505285 - 0.94407948
    # 0.79040959 - 0.47464138 - 1.14479875
    # - 0.6842171]




    #reachability
    # robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    # robot_s_meshmodel.attach_to(base)
    # for angle_x in np.linspace(3, 6, 8):
    #     for angle_y in np.linspace(-1, 1, 8):
    #         for angle_z in np.linspace(-0.5, 0, 8):
    #             tgt_pos = np.array([angle_x, angle_y, angle_z])
    #             print(tgt_pos)
    #             tgt_rotmat_x = rm.rotmat_from_axangle([1, 0, 0], -math.pi * 1 / 2)
    #             tgt_rotmat_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 1 / 2)
    #             tgt_rotmat_z = rm.rotmat_from_axangle([0, 0, 1], math.pi * 1 / 2)
    #             tgt_rotmat=np.dot(tgt_rotmat_x,tgt_rotmat_y)
    #             component_name = 'arm'
    #             jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    #             if jnt_values is None:
    #                 gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat,alpha=np.array([0.1,0.1,0.1]) ).attach_to(base)
    #             if jnt_values is not None:
    #                 # 展示末端位姿
    #                 gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat,).attach_to(base)
    #                 print('jnt_values',jnt_values)
    #                 robot_s.fk(component_name, jnt_values=jnt_values)
    #                 robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    #                 robot_s_meshmodel.attach_to(base)
    # base.run()
    # 得到的可用数据
    # [4.     0.25 - 0.125]
    # jnt_values[1.43522949
    # 0.23448567
    # 0.98138471 - 0.37366154 - 0.88240979 - 0.30891181
    # 1.50833472]
    # [4.   0.25 0.]
    # jnt_values[1.39746284
    # 0.20260791
    # 0.90022201 - 0.24676858 - 0.64252558 - 0.19822197
    # 1.33265238]
    # [4.     0.125 - 0.875]
    # jnt_values[2.24361398
    # 0.16691688
    # 1.49910806 - 1.36193624 - 1.4002439 - 1.35295734
    # 1.60627507]
    # [4.     0.125 - 0.75]
    # jnt_values[1.93808255
    # 0.21541313
    # 1.36379067 - 1.08125107 - 1.32714422 - 1.04934786
    # 1.65932326]

    #rrt静态模型
    #障碍物列表
    obstacle_list =[]
    #刀具
    test_cutter = cm.CollisionModel("./meshes/tbm_cutter_test.stl")

    test_cutter.show_cdprimit()
    print(test_cutter)
    test_cutter.set_pos(np.array([4.9, 0.23, -1]))
    # test_cutter_rotmat_x = rm.rotmat_from_axangle([1, 0, 0], math.pi * 1 / 2)
    # test_cutter_rotmat_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 1 / 2)
    # test_cutter_rotmat_z = rm.rotmat_from_axangle([0, 0, 1], math.pi * 1 / 2)
    # test_cutter_rotmat = np.dot(test_cutter_rotmat_x, test_cutter_rotmat_y)
    # test_cutter.set_rotmat(test_cutter_rotmat)
    test_cutter.set_rgba([.5, .7, .5, 1])
    test_cutter.attach_to(base)
    obstacle_list.append(test_cutter)
    print(obstacle_list[0])

    # base.run()
    # 目标位姿
    # 逆解

    component_name = 'arm'
    # 末端位姿绕x轴、y轴、z轴转的角度
    tgt_rotmat_x = rm.rotmat_from_axangle([1, 0, 0], -math.pi *0.5)
    tgt_rotmat_y = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 0.1)
    tgt_rotmat_z = rm.rotmat_from_axangle([0, 0, 1], math.pi * 0)
    # 设置末端位置和姿态
    tgt_pos = np.array([3.2445, 0.    , 0.4135])
    tgt_rotmat = np.dot(tgt_rotmat_y,np.dot(tgt_rotmat_x, rot))

    # tgt_pos = np.array([4., 0, 0.34])
    # tgt_rotmat = tgt_rotmat_x
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat,length=0.6, thickness=0.05).attach_to(base)
    # base.run()
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    # # 一次逆解直接实现模型
    robot_s.fk(component_name, jnt_values=jnt_values)
    print(jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_s_meshmodel.show_cdprimit()
    robot_s_meshmodel.attach_to(base)
    base.run()
    # rrt计算
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name='hnd',
                             start_conf=np.array([0, 0, 0, 0, 0, 0, 0]),
                             goal_conf=jnt_values,
                             #tgt_pos = np.array([3. ,  1. ,- 0.42857143])
                             #goal_conf=np.array([1.22784139,- 0.00505285, - 0.9440794,0.79040959 ,- 0.47464138 ,- 1.14479875,- 0.6842171]),
                             obstacle_list=[test_cutter],
                             ext_dist=0.1,
                             smoothing_iterations=150,
                             max_time=300)
    # print(path)
    # print(len(path))
    counter = 0
    robot_attached_list = []

    for counter in range(0, len(path)):

        pose = path[counter]
        # print('pose',pose)
        robot_s.fk(component_name, pose)
        print(robot_s.is_collided([]))
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        robot_s.show_cdprimit()
        counter = counter + 1



    # 将path导出


    # 定义数据
    data = path

    # 将数据写入 CSV 文件
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    base.run()


