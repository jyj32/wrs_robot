import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import modeling.geometric_model as gm
import modeling.model_collection as mc

class TBMArm(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='tbm', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        #cdprimitive的种类['box',  'surface_balls',  'cylinder','polygons', 'point_cloud', 'user_defined']
        #cdmesh的种类['aabb',  'obb', 'convex_hull', 'triangles']

        self.jlc = jl.JLChain(pos=pos,
                              rotmat=rotmat,
                              homeconf=homeconf,
                              cdprimitive_type='box',
                              cdmesh_type='obb',
                              name=name)
        # 7 joints, n_jnts = 7+2, n_links = 7+1

        # 1 base slide
        # self.jlc.jnts[1]['loc_pos'] = np.array([.555-x,- 0.13-x, 0.15+x])
        self.jlc.jnts[1]['loc_pos'] = np.array([1,-0.5,1.25])
        self.jlc.jnts[1]['type'] = 'prismatic'
        self.jlc.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[1]['motion_rng'] = [0, 1.5]

        self.jlc.jnts[2]['loc_pos'] = np.array([0, 0, 0.2135])
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['motion_rng'] = [-math.radians(30), math.radians(30)]

        self.jlc.jnts[3]['loc_pos'] = np.array([0.58, .0, .0])
        self.jlc.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[3]['motion_rng'] = [-math.radians(180), math.radians(180)]

        self.jlc.jnts[4]['loc_pos'] = np.array([.405, .0, .0])
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[4]['motion_rng'] = [-math.radians(90), math.radians(90)]

        self.jlc.jnts[5]['loc_pos'] = np.array([0.5565, .0, .0])
        self.jlc.jnts[5]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[5]['motion_rng'] = [-math.radians(115), math.radians(115)]

        self.jlc.jnts[6]['loc_pos'] = np.array([.559, .0, .0])
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[6]['motion_rng'] = [-math.radians(180), math.radians(180)]

        self.jlc.jnts[7]['loc_pos'] = np.array([.2190, .0, .0])
        self.jlc.jnts[7]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[7]['motion_rng'] = [-math.radians(360), math.radians(360)]

        # links 未加入质量和质心
        # self.jlc.lnks[0]['name'] = "sliderbase"
        # self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        # self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "real_tbm_base_plate.stl")
        # self.jlc.lnks[0]['rgba'] = [.35, .35, .35, 0.5]

        # self.jlc.lnks[0]['loc_pos'] = np.array([0.6, 0, -0.06])
        self.jlc.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        # self.jlc.lnks[0]['mesh_file'] = gm.gen_box(extent=[5, 0.4, 0.05])
        self.jlc.lnks[0]['mesh_file'] = gm.gen_box(extent=[0.5, 0.4, 0.05])
        self.jlc.lnks[0]['rgba'] = [.35, .35, .35, 0]

        self.jlc.lnks[1]['name'] = "base"
        self.jlc.lnks[1]['loc_pos'] = np.array([0,0,0])
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrsbase.stl")
        self.jlc.lnks[1]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[2]['name'] = "j1"
        self.jlc.lnks[2]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink1.stl")
        self.jlc.lnks[2]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[3]['name'] = "j2"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink2.stl")
        self.jlc.lnks[3]['rgba'] = [.77, .77, .60, 1]

        self.jlc.lnks[4]['name'] = "j3"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink3.stl")
        self.jlc.lnks[4]['rgba'] = [.35, .35, .35, 1]

        self.jlc.lnks[5]['name'] = "j4"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink4.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[6]['name'] = "j5"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink5.stl")
        self.jlc.lnks[6]['rgba'] = [.77, .77, .60, 1]
        #
        self.jlc.lnks[7]['name'] = "j6"
        self.jlc.lnks[7]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[7]['mesh_file'] = None
        self.jlc.lnks[7]['rgba'] = [.5, .5, .5, 1]

        self.jlc.reinitialize()
        # collision detection
        if enable_cc:
            self.enable_cc()
        # cd meshes collection for precise collision checking
        self.cdmesh_collection = mc.ModelCollection()


    # def enable_cc(self):
    #     super().enable_cc()
    #     self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6,])
    #     activelist = [self.jlc.lnks[0],
    #                   self.jlc.lnks[1],
    #                   self.jlc.lnks[2],
    #                   self.jlc.lnks[3],
    #                   self.jlc.lnks[4],
    #                   self.jlc.lnks[5],
    #                   self.jlc.lnks[6],
    #
    #                   ]
    #     self.cc.set_active_cdlnks(activelist)
    #     fromlist = [self.jlc.lnks[0]]
    #     intolist = [self.jlc.lnks[2],
    #                 self.jlc.lnks[3],
    #                 self.jlc.lnks[4],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6],
    #
    #                 ]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[1]]
    #     intolist = [self.jlc.lnks[3],
    #                 self.jlc.lnks[4],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6],
    #
    #                 ]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[2]]
    #     intolist = [self.jlc.lnks[4],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6],
    #                 ]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[3]]
    #     intolist = [self.jlc.lnks[5],
    #                 self.jlc.lnks[6],
    #
    #                 ]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[4]]
    #     intolist = [self.jlc.lnks[6],
    #
    #                 ]
    #     self.cc.set_cdpair(fromlist, intolist)




if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3.7,-4,1.7],lookat_pos=[1.5,0,.3])
    gm.gen_frame().attach_to(base)
    manipulator_instance = TBMArm(enable_cc=True)
    robot_s_stickmodel = manipulator_instance.gen_stickmodel(toggle_jntscs=True)
    robot_s_stickmodel.attach_to(base)

    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.show_cdprimit()


    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)

    base.run()

    #改零点坐标的位置
   # self.jlc.lnks[0]['loc_pos'] = np.array([0.1, 0, -0.06])




