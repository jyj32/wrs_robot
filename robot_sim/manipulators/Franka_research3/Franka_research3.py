import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))#根目录错误时
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import time

start = time.time()
class Franka(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='Franka_research3', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # 7 joints(7+1)
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.141])  # 平移矩阵
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])  # 旋转轴

        self.jlc.jnts[2]['loc_pos'] = np.array([-0.000009, -0.000500, 0.192025])  # 平移矩阵
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-np.pi / 2, -20/180 * np.pi, 0)  # 旋转矩阵
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])  # 旋转轴

        self.jlc.jnts[3]['loc_pos'] = np.array([0, -0.192974, 0])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(np.pi / 2, 0, 0)# 旋转矩阵
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[4]['loc_pos'] = np.array([0.082504, -0.0005, 0.122952])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(np.pi / 2, 145/180 * np.pi, 0)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[5]['loc_pos'] = np.array([-0.082504, 0.123952 , -0.000500 ])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(-np.pi / 2, 0, 0)
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[6]['loc_pos'] = np.array([0, 0.016401, 0.260])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(np.pi / 2, -125/180 * np.pi, 0)
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])

        self.jlc.jnts[7]['loc_pos'] = np.array([0.088012, -0.05045, 0.016429])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(np.pi/2, 0, 0)

        self.jlc.jnts[8]['loc_pos'] = np.array([0, 0, 0.05665])
        self.jlc.jnts[8]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)

        # 8 links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.array([.0, 0.0, .0])   # 碰撞体的位置
        self.jlc.lnks[0]['mass'] = 2.0
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "link0.STL")
        self.jlc.lnks[0]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[1]['name'] = "shoulder1"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.0, -.02, .0])  # 质心位置
        self.jlc.lnks[1]['mass'] = 1.95
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link1.STL")
        self.jlc.lnks[1]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[2]['name'] = "shoulder2"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.jlc.lnks[2]['mass'] = 3.42
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link2.STL")
        self.jlc.lnks[2]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[3]['name'] = "elbow1"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.jlc.lnks[3]['mass'] = 1.437
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link3.STL")
        self.jlc.lnks[3]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[4]['name'] = "elbow2"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[4]['mass'] = 0.871
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link4.STL")
        self.jlc.lnks[4]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[5]['name'] = "lowerarm"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.jlc.lnks[5]['mass'] = 0.8
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link5.STL")
        self.jlc.lnks[5]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[6]['name'] = "wrist1"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, -0.02])
        self.jlc.lnks[6]['mass'] = 0.8
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link6.STL")
        self.jlc.lnks[6]['rgba'] = [.7,.7,.7, 1]

        self.jlc.lnks[7]['name'] = "flange"
        self.jlc.lnks[7]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[7]['com'] = np.array([.0, .0, -0.02])
        self.jlc.lnks[7]['mass'] = 0.8
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "link7.STL")
        self.jlc.lnks[7]['rgba'] = [.7,.7,.7, 1]

        self.jlc.reinitialize()
        # collision checker
        if enable_cc:
            super().enable_cc() # 碰撞检测

    def enable_cc(self):    # 碰撞体

        super().enable_cc()
        # 添加碰撞体
        self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6, 7]) # 机器人 self.jlc 中的链接0到链接7添加到碰撞检测列表中

        # 激活碰撞体
        activelist = [self.jlc.lnks[0],
                      self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6],
                      self.jlc.lnks[7]]
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


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = Franka(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    # manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    end = time.time()
    print('程序运行时间为: %s Seconds' % (end - start))
    # tic = time.time()
    print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)
    manipulator_meshmodel.show_cdprimit()   # 显示碰撞体
    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # gm.GeometricModel("./meshes/base.dae").attach_to(base)
    # gm.gen_frame().attach_to(base)
    base.run()
