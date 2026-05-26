import robot_sim.robots.ur7e.ur7e as ur7
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.model_collection as mc
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_con.ur.ur7_dh50_rtde as ur7con
import numpy as np
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd


if __name__ == '__main__':
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    rbt_s = ur7.UR7E()
    gripper_s = hnd.Dh50()

    gm.gen_frame().attach_to(base)
    rbt_s.gen_meshmodel().attach_to(base)

    test_pos = np.array([-1.34929917e-04, 5.67840796e-01, 3.32778857e-01 - 0.128])
    # test_pos = np.array([0.325, 0.725, 0.14])
    test_rot = rm.rotmat_from_axangle([0, 1, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], -np.pi/2))
    # test_rot = np.array([[-1, 0, 1.2246e-16],
    #        [0, 1, 0],
    #        [-1.2246e-16, 0, -1]])
    # test_rot = rm.rotmat_from_axangle([0, 1, 0], np.pi) @ rm.rotmat_from_axangle([1, 0, 0], -np.pi / 4)
    gm.gen_frame(pos=test_pos, rotmat=test_rot).attach_to(base)
    test_jnt_tracik = rbt_s.tracik(tgt_pos=test_pos, tgt_rotmat=test_rot, solver_type='Manip2')
    test_jnt_ik = rbt_s.ik('arm', test_pos, test_rot)
    rbt_s.fk('arm', test_jnt_tracik)
    rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.5]).attach_to(base)
    rbt_s.fk('arm', test_jnt_ik)
    rbt_s.gen_meshmodel(rgba=[0, 0, 1, 0.5]).attach_to(base)

    rbt_s.fk('arm', np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi]))
    rbt_s.gen_meshmodel(rgba=[1, 0, 0, 0.5]).attach_to(base)
    # base.run()

    rbt_r = ur7con.UR5Ag95X_RTDE(robot_ip='192.168.125.30',
                                 gp_port='COM5')
    rbt_r.move_jnts(np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi]), 0.5)

    rbt_r.close_to_dh(50)
    rbt_r.open_gripper_dh()
    rbt_r.close_gripper_dh()
    rbt_r.move_jnts(test_jnt_ik, 0.5)
    current_jnts = rbt_r.get_jnt_values()
    rbt_s.fk('arm', current_jnts)
    print(rbt_s.hnd.jaw_center_pos)
    print(rbt_s.get_gl_tcp('arm'))

    # [-1.34929917e-04,  5.67840796e-01,  3.32778857e-01]
    gm.gen_sphere(np.array([-1.34929917e-04,  5.67840796e-01,  3.32778857e-01-0.128]), 0.01).attach_to(base)

    rbt_s.gen_meshmodel().attach_to(base)

    # rbt_r.move_jnts(test_jnt_ik, vel=0.5)

    base.run()
