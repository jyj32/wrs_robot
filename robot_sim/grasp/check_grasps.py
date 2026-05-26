"""
用来检查抓取的姿态
"""
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
#import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.end_effectors.gripper.ag145.ag145 as dh #使用AG145的夹爪

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)

gripper_s = dh.Ag145()
objcm_name = "123"
obj = cm.CollisionModel(f"{objcm_name}.STL")
obj.set_rgba([.9, .75, 0.35, 1])

rotmat_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
rotmat_x_180 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])#由于试管建模时是头朝下，导入需要调转方向
obj.set_rotmat(rotmat_x_90)#调转试管脑袋
obj.attach_to(base)
# 抓取信息列表
grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='123_AG145_5.pickle')
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    #gripper_s.gen_meshmodel(rgba=[0,1,0,1]).attach_to(base)
    gripper_s.gen_meshmodel().attach_to(base)
    # break   #取消注释的话则只显示第一张抓取办法

base.run()