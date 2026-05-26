"""
将方块从C位置移动到D位置
"""
import os
import copy
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.frank_research3.frank_research3 as dh
import robot_sim.robots.Franka_research3.Franka_research3 as Franka
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr

if __name__ == '__main__':
    rotmat_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])  # 绕X轴90度旋转矩阵
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])  # 创建3D世界并设置相机
    gm.gen_frame().attach_to(base)  # 坐标系

    rbt_s = Franka.Franka_research3()  # 机器人
    rbt_s.hnd.jaw_to(0.0)  # 机器人手爪0张开
    rbt_s.gen_meshmodel().attach_to(base)  # 将机器人模型添加到场景
    manipulator_name = "arm"  # 机械臂名
    # start_conf = rbt_s.get_jnt_values(manipulator_name)
    start_conf = np.array([0, 0, 0, 0, 0, 0, 0])  # 起始关节角度
    # print(start_conf)
    hand_name = "hnd"  # 手爪
    # object放置初始物体
    objcm_name = "cube50"  # 物体
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    stl_path = os.path.join(current_dir, "cube50.STL")
    # 然后使用这个路径
    obj = cm.CollisionModel(stl_path)  # 创建碰撞模型
    # 初始物体
    obj.set_rgba([.9, .75, .35, 1])  # 颜色
    obj.set_pos(np.array([.4, -.4, 0.05]))  # 开始位置
    obj.set_rotmat(np.eye(3))  # 开始姿态
    obj.show_localframe()
    obj.attach_to(base)

    # object_goal目标位置
    obj_goal = cm.CollisionModel(stl_path)
    obj_goal.set_rgba([1, 1, 1, 1])  # 颜色
    obj_goal.set_pos(np.array([.4, .4, 0.05]))  # 目标位置
    obj_goal.set_rotmat(np.eye(3))  # 目标姿态
    obj_goal.attach_to(base)

    # base.run()

    gripper_s = dh.frank_research3()  # 创建手爪实例
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=current_dir,
                                           file_name='cube50_Franka.pickle')  # 从pickle文件中加载预计算抓取姿态
    # print(grasp_info_list)
    # base.run()
    # for grasp_info in grasp_info_list:
    #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    #     jaw_center_pos = jaw_center_pos + np.array([0.4, -0.4, 0.35])
    #     jaw_center_rotmat = jaw_center_rotmat@rotmat_x_90
    #     gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    #
    #     gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    #     print(jaw_width)
    #     rbt_s.jaw_to(jaw_width)
    #     test_conf = rbt_s.ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat)
    #     rbt_s.fk('arm', test_conf)
    #     rbt_s.gen_meshmodel().attach_to(base)
    #     base.run()

    start_pos = obj.get_pos()  # 开始位置
    start_rotmat = obj.get_rotmat()  # 开始姿态
    # ===================================
    start_homo = rm.homomat_from_posrot(start_pos, start_rotmat)  # 将物体的位置和旋转矩阵转换为一个齐次变换矩阵
    jnts_list = []
    for grasp_info in grasp_info_list:  # 抓取姿态提取
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info  #
        jnts = rbt_s.ik(component_name="arm", tgt_pos=rm.homomat_transform_points(start_homo, jaw_center_pos),
                        tgt_rotmat=start_rotmat.dot(jaw_center_rotmat))  # 机器人名，位置，角度，ik求解
        jnts_list.append(jnts)  # IK求解结果（关节角度或None）添加到列表

    # print("开始位置的IK解：",  jnts_list)

    rrtc_planner = rrtc.RRTConnect(rbt_s)  # 创建RRT-Connect路径规划器实例，用于后续的运动规划

    goal_pos = obj_goal.get_pos()  # 目标位置
    goal_rotmat = obj.get_rotmat()  # 目标姿态

    # goal_jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    # rbt_s.fk(component_name="arm", jnt_values=goal_jnt_values)
    # rbt_s.gen_meshmodel().attach_to(base)
    # 创建RRT-Connect规划器和抓取放置规划器
    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)
    # 计算物体的起始和目标齐次变换矩阵
    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
    # conf_list：机器人手臂的关节角度序列（每个时刻的姿势）
    # jawwidth_list：手爪宽度序列（什么时候张开、什么时候闭合）
    # objpose_list：物体的姿态序列（被抓取后随机器人移动的轨迹
    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=None,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
                                        approach_distance_list=[.1] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 1])],
                                        depart_distance_list=[.1] * 2)  # 生成完整的抓取-放置运动轨迹
    robot_attached_list = []
    object_attached_list = []
    counter = [0]


    def update(robot_s,
               object_box,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):

        if robot_path is None:
            print("警告：路径生成失败，无法继续执行")
            return  # 或其他错误处理逻辑
        if counter[0] >= len(robot_path):  # 循环播放轨迹
            counter[0] = 0
        if len(robot_attached_list) != 0:  # 清除上一帧的模型
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        # 更新当前帧的机器人姿势
        pose = robot_path[counter[0]]
        robot_s.fk(manipulator_name, pose)  # 正运动学求解
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])  # 设置手爪宽度
        robot_meshmodel = robot_s.gen_meshmodel()  # 生成并添加新的机器人模型
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        # 更新物体位置
        obj_pose = obj_path[counter[0]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])  # 设置抓取后的物体颜色为红色
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1  # 更新计数器
        return task.again  # 继续执行任务


    # 设置定时任务，每0.01秒更新一帧
    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[rbt_s,
                                     obj,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()  # 运行主循环
