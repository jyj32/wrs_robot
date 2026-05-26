""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20220811osaka

"""
import os
import numpy as np
from trac_ik import TracIK

if __name__ == '__main__':


    urdf_path = r'F:\Study\point cloud\wrs-qiu\wrs-qiu\0000_regrasp_for_fastener\gofa5.urdf'

    # yumi_rgt_arm_iksolver = TracIK(base_link_name="yumi_body",
    #                                tip_link_name="yumi_link_7_r",
    #                                urdf_path=urdf_path, )
    gofa5_arm_iksolver = TracIK(base_link_name="base_link",
                                tip_link_name="link_6",
                                urdf_path=urdf_path)
    # yumi_lft_arm_iksolver = TracIK(base_link_name="yumi_body",
    #                                tip_link_name="yumi_link_7_l",
    #                                urdf_path=urdf_path, )
    # seed_jnt = np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
    #                      0.6981317, 0.])
    seed_jnt = np.array([1.71127459e-01, 1.82249335e-01, 5.43805745e-01, -1.58542253e-05,
                         8.44677978e-01, 1.71138133e-01])
    # tgt_pos = np.array([.3, -.4, .1])
    tgt_pos = np.array([0.733, 0.063, 0.18])
    # tgt_rotmat = np.array([[0.5, 0., 0.8660254],
    #                        [0., 1., 0.],
    #                        [-0.8660254, 0., 0.5]])
    tgt_rotmat = np.array([[-0.99652, -0.083406, 0],
                         [-0.083406, 0.99652, 0],
                         [0, 0, -1]])
    # result = yumi_rgt_arm_iksolver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt)
    result = gofa5_arm_iksolver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt)
    print("The ik solution is", result)

    pos_fk, rot_fk = gofa5_arm_iksolver.fk(result)
    print("The fk result is", pos_fk, rot_fk)
