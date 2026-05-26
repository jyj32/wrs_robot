import numpy as np
import robot_con.gofa_con.gofa_con as gofa_con
import robot_sim.robots.gofa5.gofa5 as gf5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm


if __name__ == "__main__":
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame(length=0.2).attach_to(base)
    move_rbt = True
    rbt_s = gf5.GOFA5()
    rbt_s.gen_meshmodel().attach_to(base)
    if move_rbt:
        rbt_r = gofa_con.GoFaArmController()


    def mov_rbt_to_pos_and_rot(pos, rot, seed=None, draw=False):
        conf = rbt_s.ik('arm', pos, rot, seed_jnt_values=seed)
        # conf = rbt_s.tracik(tgt_pos=pos,
        #                     tgt_rotmat=rot,
        #                     seed_jnt_values=seed)
        if conf is None:
            return None
        rbt_s.fk('arm', conf)
        if rbt_s.is_collided():
            print('collision detected')
            return None
        if draw:
            print(f'jnt value: {conf}')
            print('ik and fk success')
            rbt_s.fk('arm', conf)
            rbt_s.gen_meshmodel().attach_to(base)
        if move_rbt:
            rbt_r.move_j(conf)
        return conf


    # def is_rotation_matrix(R):
    #     should_be_identity = R.T @ R
    #     I = np.identity(3)
    #     return np.allclose(should_be_identity, I, atol=1e-6) and np.isclose(np.linalg.det(R), 1.0)

    # if move_rbt:
    #     rbt_r.move_j(np.array([0., 0., 0., 0., 0., 0.]))
    # base.run()
    # nut_pos = np.array([0.75-0.013, -0.4+0.0125, 0.055])

    center_pos = np.array([0.733, 0.063, 0.18])
    center_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], np.pi/2))

    # test_rot = np.array([[-0.99652, -0.083406, 0],
    #                      [-0.083406, 0.99652, 0],
    #                      [0, 0, -1]])

    # seed_jnt = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
    #                      8.44677978e-01, 1.71138133e-01])

    # gm.gen_frame(pos=np.array([0.733, 0.063, 0.18]), rotmat=test_rot).attach_to(base)

    # for angle in np.linspace(0, 360, 361):
    #     rot = rm.rotmat_from_axangle([1, 0, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], np.pi*angle/180))
    #     conf = mov_rbt_to_pos_and_rot(pos=center_pos, rot=rot, seed=seed_jnt)
    #     if conf is None:
    #         gm.gen_frame(pos=center_pos, rotmat=rm.rotmat_from_axangle([1, 0, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], np.pi*angle/180))).attach_to(base)
    #         print(angle)

    # print(test_rot)
    # print(is_rotation_matrix(test_rot))
    rbt_r.move_j(np.array([0., 0., 0., 0., 0., 0.]))
    # mov_rbt_to_pos_and_rot(center_pos, center_rot, draw=True)
    base.run()