import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, Point3, CollisionCapsule

class UR7E(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='ur7e', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        jnt_safemargin = math.pi / 18.0
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.163])
        self.jlc.jnts[1]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]

        self.jlc.jnts[2]['loc_pos'] = np.array([0, 0.138, 0])
        self.jlc.jnts[2]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(.0, math.pi / 2.0, .0)
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])

        self.jlc.jnts[3]['loc_pos'] = np.array([0, -.131, .425])
        self.jlc.jnts[3]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 1, 0])

        self.jlc.jnts[4]['loc_pos'] = np.array([.0, .0, 0.392])
        self.jlc.jnts[4]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(.0, math.pi/2.0, 0)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 1, 0])

        self.jlc.jnts[5]['loc_pos'] = np.array([0, .127, 0])
        self.jlc.jnts[5]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[6]['loc_pos'] = np.array([0, 0, .100])
        self.jlc.jnts[6]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 1, 0])

        self.jlc.jnts[7]['loc_pos'] = np.array([0, .100, 0])
        self.jlc.jnts[7]['motion_rng'] = [-2*math.pi + jnt_safemargin, 2*math.pi  - jnt_safemargin]
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2.0, 0, 0)
        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mass'] = 2.0
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.dae")
        self.jlc.lnks[0]['rgba'] = [.5,.5,.5, 1]
        self.jlc.lnks[1]['name'] = "shoulder"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.jlc.lnks[1]['mass'] = 1.95
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "shoulder.dae")
        self.jlc.lnks[1]['rgba'] = [.1,.3,.5, 1]
        self.jlc.lnks[2]['name'] = "upperarm"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.jlc.lnks[2]['mass'] = 3.42
        # 使用自定义碰撞体
        self.jlc.lnks[2]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "upperarm.dae"),
                cdprimit_type="user_defined",
                userdefined_cdprimitive_fn=self._upperarm_cdnp_lnks2,
                expand_radius=.001)
        self.jlc.lnks[2]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[3]['name'] = "forearm"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.jlc.lnks[3]['mass'] = 1.437
        # self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "forearm.dae")
        self.jlc.lnks[3]['mesh_file'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "forearm.dae"),
                cdprimit_type="user_defined",
                userdefined_cdprimitive_fn=self._upperarm_cdnp_lnks3,
                expand_radius=.001)
        self.jlc.lnks[3]['rgba'] = [.35,.35,.35, 1]
        self.jlc.lnks[4]['name'] = "wrist1"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[4]['mass'] = 0.871
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist1.dae")
        self.jlc.lnks[4]['rgba'] = [.7,.7,.7, 1]
        self.jlc.lnks[5]['name'] = "wrist2"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[5]['mass'] = 0.8
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist2.dae")
        self.jlc.lnks[5]['rgba'] = [.1,.3,.5, 1]
        self.jlc.lnks[6]['name'] = "wrist3"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, -0.02])
        self.jlc.lnks[6]['mass'] = 0.8
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "wrist3.dae")
        self.jlc.lnks[6]['rgba'] = [.5,.5,.5, 1]
        self.jlc.reinitialize()
        # collision checker
        if enable_cc:
            super().enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[0],
                    self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[3],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[4],
                    self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)

    def _upperarm_cdnp_lnks2(self, name, radius):

        collision_node = CollisionNode(name)

        # 长方体1：肩部连接段
        # 参数：Point3(x,y,z) 中心位置, x/y/z 分别是半长/半宽/半高
        box1 = CollisionBox(Point3(0.0, 0.003, 0.01),
                            x=0.06 + radius,
                            y=0.07 + radius,
                            z=0.07 + radius)
        collision_node.addSolid(box1)

        # 长方体2：上臂主体段（最长的部分）
        box2 = CollisionBox(Point3(0.0, 0, 0.25),
                            x=0.042 + radius,
                            y=0.042 + radius,
                            z=0.18 + radius)
        collision_node.addSolid(box2)
        #
        # # 长方体3：肘部连接段
        box3 = CollisionBox(Point3(0.0, 0.003, 0.411),
                            x=0.06 + radius,
                            y=0.07 + radius,
                            z=0.073 + radius)
        collision_node.addSolid(box3)

        return collision_node

    def _upperarm_cdnp_lnks3(self, name, radius):
        collision_node = CollisionNode(name)

        # 长方体1：肩部连接段
        # 参数：Point3(x,y,z) 中心位置, x/y/z 分别是半长/半宽/半高
        box1 = CollisionBox(Point3(0.0, 0.003, 0.01),
                            x=0.057 + radius,
                            y=0.051 + radius,
                            z=0.07 + radius)
        collision_node.addSolid(box1)

        # 长方体2：上臂主体段（最长的部分）
        box2 = CollisionBox(Point3(0.0, 0, 0.215),
                            x=0.039 + radius,
                            y=0.039 + radius,
                            z=0.135 + radius)
        collision_node.addSolid(box2)
        #
        # # 长方体3：肘部连接段
        box3 = CollisionBox(Point3(0.0, 0.003, 0.39),
                            x=0.038 + radius,
                            y=0.061 + radius,
                            z=0.041 + radius)
        collision_node.addSolid(box3)

        return collision_node


    # def _upperarm_cdnp_lnks2(self, name, radius):
    #     """
    #     为 UR7E 上臂创建自定义碰撞体（3个圆柱体组合）
    #     坐标系原点：连杆2的局部坐标系原点（即关节2的位置）
    #     :param name: 碰撞节点名称
    #     :param radius: 扩展半径（用于增加安全边界）
    #     :return: CollisionNode
    #     """
    #     collision_node = CollisionNode(name)
    #
    #     # 使用 CollisionCapsule（胶囊体）来近似圆柱体
    #     # CollisionCapsule 由两个端点和半径定义，更适合表示臂杆
    #
    #     # 圆柱体1：肩部到上臂中段
    #     # 参数：Point3(x1,y1,z1) 起点, Point3(x2,y2,z2) 终点, radius 半径
    #     capsule1 = CollisionCapsule(Point3(0.0, 0, 0.12),
    #                                 Point3(0.0, 0, 0.4),
    #                                 0.05 + radius)
    #     collision_node.addSolid(capsule1)
    #
    #     # 圆柱体2：上臂中段主体
    #     capsule2 = CollisionCapsule(Point3(0.0, -0.01, 0.01),
    #                                 Point3(0., 0.01, 0.01),
    #                                 0.085 + radius)
    #     collision_node.addSolid(capsule2)
    #
    #
    #     capsule3 = CollisionCapsule(Point3(0.0, 0, 0.04),
    #                                 Point3(0.0, 0, 0.05),
    #                                 0.07 + radius)
    #     collision_node.addSolid(capsule3)
    #
    #     capsule4 = CollisionCapsule(Point3(0.0, -0.03, 0.41),
    #                                 Point3(0.0, 0.005, 0.41),
    #                                 0.087 + radius)
    #     collision_node.addSolid(capsule4)
    #
    #     capsule5 = CollisionCapsule(Point3(0.0, 0, 0.37),
    #                                 Point3(0.0, 0, 0.39),
    #                                 0.07 + radius)
    #     collision_node.addSolid(capsule5)
    #     return collision_node

    # def _upperarm_cdnp(self, name, radius):
    #     """
    #     为 UR7E 上臂创建自定义碰撞体（3个圆柱体组合）
    #     坐标系原点：连杆2的局部坐标系原点（即关节2的位置）
    #     :param name: 碰撞节点名称
    #     :param radius: 扩展半径（用于增加安全边界）
    #     :return: CollisionNode
    #     """
    #     collision_node = CollisionNode(name)
    #
    #     # 使用 CollisionCapsule（胶囊体）来近似圆柱体
    #     # CollisionCapsule 由两个端点和半径定义，更适合表示臂杆
    #
    #     # 圆柱体1：肩部到上臂中段
    #     # 参数：Point3(x1,y1,z1) 起点, Point3(x2,y2,z2) 终点, radius 半径
    #     capsule1 = CollisionCapsule(Point3(0.0, 0, 0.12),
    #                                 Point3(0.0, 0, 0.4),
    #                                 0.05 + radius)
    #     collision_node.addSolid(capsule1)
    #
    #     # 圆柱体2：上臂中段主体
    #     capsule2 = CollisionCapsule(Point3(0.0, -0.01, 0.01),
    #                                 Point3(0., 0.01, 0.01),
    #                                 0.085 + radius)
    #     collision_node.addSolid(capsule2)
    #
    #
    #     capsule3 = CollisionCapsule(Point3(0.0, 0, 0.04),
    #                                 Point3(0.0, 0, 0.05),
    #                                 0.07 + radius)
    #     collision_node.addSolid(capsule3)
    #
    #     capsule4 = CollisionCapsule(Point3(0.0, -0.03, 0.41),
    #                                 Point3(0.0, 0.005, 0.41),
    #                                 0.087 + radius)
    #     collision_node.addSolid(capsule4)
    #
    #     capsule5 = CollisionCapsule(Point3(0.0, 0, 0.37),
    #                                 Point3(0.0, 0, 0.39),
    #                                 0.07 + radius)
    #     collision_node.addSolid(capsule5)
    #     return collision_node


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = UR7E(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    manipulator_meshmodel.show_cdprimit()
    base.run()
    # manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)


    # conf = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # manipulator_instance.fk(conf)
    # manipulator_instance.gen_meshmodel().attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_meshmodel().attach_to(base)

    base.run()
    # tic = time.time()
    # print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # gm.GeometricModel("./meshes/base.dae").attach_to(base)
    # gm.gen_frame().attach_to(base)
    base.run()