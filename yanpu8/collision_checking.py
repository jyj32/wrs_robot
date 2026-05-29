import math

import time

import robot_sim.robots.ur7e.ur7ewithoutmachine as cbt
import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as cm
import modeling.geometric_model as gm
import basis.robot_math as rm
import path_plan as pp
from config import CONFIG_U1,CONFIG_U625,CONFIG_U1_cc_xipan,CONFIG_U625_cc_xipan
import motion.probabilistic.rrt_connect as rrtc

# 机器人碰撞检测
if __name__ == '__main__':
    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    ur7e_s = cbt.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),enable_cc=True)  # 仿真机器人，可以以仿真机器人为基准
    # ur7e_s.gen_meshmodel().attach_to(base)

    # 障碍物模型
    obstacle_list = ur7e_s.get_obstacle_list(base,True)

    # 1.检测某个关节点是否碰撞
    # jnts = np.array([ 2.6971895077412027, -3.190632673671864, 1.2576356005448501, 0.36220098055745203, -1.5707960937890162, 0.6590745034072273])
    # ur7e_s.fk("arm", jnts)
    # ur7e_s.jaw_to('hnd', 0.02) # 手爪闭合再检测是否碰撞
    # collision_info = ur7e_s.is_collided(obstacle_list)
    # print(collision_info)
    # ur7e_s.gen_meshmodel().attach_to(base)

    # 2.加上ik求解
    '''
    '''

    pos_0 = np.array([ 0.8296444230918749, -0.24276065206145675, 1.256209829843713])
    # wait_rot = rm.rotmat_from_axangle([0, 1, 0], math.pi)
    # wait_rot = np.array([[   -1.0000000e+00 , 0.0000000e+00 , 1.2246468e-16],
    #                      [ 0.0000000e+00 , 1.0000000e+00 , 0.0000000e+00],
    #                      [-1.2246468e-16 , 0.0000000e+00 ,-1.0000000e+00]])
    wait_rot = np.array([[  0.02313107 ,-0.9993294 ,  0.02838492],
                         [-0.83222496 ,-0.03498004 ,-0.55333355],
                         [ 0.55395539, -0.01082344 ,-0.83247599]])
    # pos_0 = CONFIG_U625['grasp']['box_center_pos']
    # wait_rot = CONFIG_U1['place']['0']['high_place_rot']
    # pos_0 = CONFIG_U1['place']['0']['high_place_pos']
    # wait_rot = CONFIG_U1['place']['-60']['place_rot']
    # seed = np.array([0.61461222,-1.37349709,1.55544195,-1.75279925,-1.57078562,0.61457421])
    # 用于U1的seed
    seed = CONFIG_U625['grasp']['box_center_conf']
    # 用于U625的seed
    # seed = CONFIG_U625['grasp']['box_center_conf']

    jnts = ur7e_s.tracik( tgt_pos=pos_0,
                           tgt_rotmat=wait_rot,
                           seed_jnt_values=seed,
                           solver_type = "Distance"
                          )
    print(list(jnts))   # 转换为列表再打印就会带逗号
    ur7e_s.fk("arm", jnts)
    collision_info = ur7e_s.is_collided(obstacle_list)
    print(collision_info)
    ur7e_s.gen_meshmodel().attach_to(base)

    # 3.检测hold是否有用
    # obj_cm = cm.CollisionModel("object/U1_2.STL")
    # wait_rot = np.dot(
    #     rm.rotmat_from_axangle([0, 1, 0], math.pi),
    #     rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # )
    # U_rot = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # pos_0 = np.array([0.5, -0.13, 0.18])
    # jnts = ur7e_s.ik("arm", pos_0, wait_rot)
    # ur7e_s.fk("arm", jnts)
    # obj_cm.set_rotmat(U_rot)  # 物体绕Z轴旋转角度
    # obj_cm.set_pos(pos_0)  # 物体在世界中的初始位置
    # rel_pos, rel_rotmat = ur7e_s.hold("arm", obj_cm, jawwidth=0)  # 抓住
    # # 此时 rel_pos, rel_rotmat 可用于后续运动规划
    # print("相对位置:", rel_pos)
    # print("相对旋转:", rel_rotmat)
    # collision_info = ur7e_s.is_collided(obstacle_list)
    # print(collision_info)
    # ur7e_s.gen_meshmodel().attach_to(base)

    # pos_1 = np.array([0.5, -0.16, 0.19])
    # jnts = ur7e_s.ik("arm", pos_1, wait_rot)
    # ur7e_s.fk("arm", jnts)
    # obj_cm.attach_to(base)
    # ur7e_s.gen_meshmodel().attach_to(base)
    # collision_info = ur7e_s.is_collided(obstacle_list)
    # print(collision_info)
    # check_jnts = np.array([1.6708  ,  -0.84991   ,   1.1234  ,   -1.8444  ,   -1.5708    ,  1.1845])

    # 4.检查路径规划
    # plan_path = pp.Path_plan(ur7e_s,obstacle_list,base)
    # wait_rot = np.dot(
    #     rm.rotmat_from_axangle([0, 1, 0], math.pi),
    #     rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # )
    # pos_0 = np.array([0.5, 0, 0.13])
    # jnts0 = ur7e_s.ik("arm", pos_0, wait_rot)
    # jnts0 = np.array([    0.61461  ,   -1.3735    ,  1.5554   ,  -1.7528   ,  -1.5708   ,  0.61457])
    # jnts0 = CONFIG_U1['abandon']['abandon_conf']
    #
    # ur7e_s.fk("arm", jnts0)
    # collision_info = ur7e_s.is_collided(obstacle_list)
    # print(collision_info)
    # ur7e_s.gen_meshmodel().attach_to(base)
    # # pos_1 = np.array([0.5, -0.25, 0.13])
    # # jnts1 = ur7e_s.ik("arm", pos_1, wait_rot)
    # # jnts1 =np.array([    1.494  ,   -1.2637    ,   1.485    , -1.7922  ,   -1.5708  ,   0.18384])
    # jnts1 = CONFIG_U1['grasp']['wait_conf']
    # ur7e_s.fk("arm", jnts1)
    # collision_info = ur7e_s.is_collided(obstacle_list)
    # print(collision_info)
    # ur7e_s.gen_meshmodel().attach_to(base)
    # path = plan_path.plan_path(jnts0,jnts1,step = 0.1)
    # print(path)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in path:
    #     ur7e_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = ur7e_s.gen_meshmodel(rgba=[0,1,0,0.5])
    #     rbt_mesh.attach_to(base)

    # 5.抓住物体的路径规划
    # plan_path = pp.Path_plan(ur7e_s,obstacle_list,base)
    # # wait_rot = np.dot(
    # #     rm.rotmat_from_axangle([0, 1, 0], math.pi),
    # #     rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    # # )
    # pos_0 = np.array([ 0.85434  ,   -0.1926  ,     1.019])
    # #
    # # jnts0 = ur7e_s.ik("arm", pos_0, wait_rot)
    # jnts0 = np.array([1.6238542055481617, -1.7875247715810334, 1.7496286503427603, -1.5329002048348261, -1.570796326045198, 0.053057878753263205])
    # # pos_0 = CONFIG_U1['grasp']['box_center_pos']
    # # jnts0 = CONFIG_U1['grasp']['box_center_conf']
    #
    # # # collision_info = ur7e_s.is_collided(obstacle_list)
    # # print(collision_info)
    # obj_cm = cm.CollisionModel("object/U625.STL")
    # U_rot = rm.rotmat_from_axangle([0, 0, 1], 0)
    # obj_cm.set_rotmat(U_rot)  # 物体绕Z轴旋转角度
    # obj_cm.set_pos(pos_0)  # 物体在世界中的初始位置
    #
    # # ur7e_s.gen_meshmodel().attach_to(base)
    # # pos_1 = np.array([0.5, -0.25, 0.13])
    # # jnts1 = ur7e_s.ik("arm", pos_1, wait_rot)
    # # jnts1 = np.array([2.29533901, -1.52839454 , 1.73042229, -1.77282405, -1.57079613 , 2.29533926])
    # jnts1 = CONFIG_U625['place']['high_place_conf']
    # ur7e_s.fk("arm", jnts0)
    # rel_pos, rel_rotmat = ur7e_s.hold("arm", obj_cm, jawwidth=0.02)  # 抓住
    # # ur7e_s.fk("arm", jnts1)
    # # collision_info = ur7e_s.is_collided(obstacle_list)
    # # print(collision_info)
    # # ur7e_s.gen_meshmodel().attach_to(base)
    # path = plan_path.plan_path(jnts0,jnts1)
    # print(path)
    # # 展示路径
    # rbt_mesh = None
    # for jnts1 in path:
    #     ur7e_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
    #     rbt_mesh = ur7e_s.gen_meshmodel(rgba=[0,1,0,0.5])
    #     rbt_mesh.attach_to(base)

    # 直线路径规划
    # plan_path = pp.Path_plan(ur7e_s, obstacle_list)
    # pos1 = np.array([0.1, 0.1, 0.9])
    # pos2 = np.array([0.4, -0.1, 1.04])
    # pos3 = np.array([0.9, -0.2, 0.9])
    # rot = rm.rotmat_from_axangle([0, 1, 0], math.pi)
    # seed = CONFIG_U1['grasp']['box_center_conf']
    # path1 = plan_path.get_line_path(seed, pos1, pos2, rot)
    # print(f'path1:{path1}')
    # path2 = plan_path.get_line_path(seed, pos2, pos3, rot)
    # print(f'path2:{path2}')
    # path3 = path1 + path2
    # print(f'path3:{path3}')
    # rrt_planner = rrtc.RRTConnect(ur7e_s)
    # start_time = time.time()
    # smoothed_path = rrt_planner._smooth_path(
    #     component_name='arm',
    #     path=path3,
    #     obstacle_list=obstacle_list,
    #     granularity=0.1,
    #     iterations=50,
    # )
    # print(time.time() - start_time)
    # print(f'smoothed_path:{smoothed_path}')
    # 机器人展示碰撞体
    ur7e_s.show_cdprimit()

    base.run()