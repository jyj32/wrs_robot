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
import robot_sim.end_effectors.gripper.dh50longfinger.dh50longfinger as dh
import robot_sim.robots.ur7e.ur7ewithoutmachine as UR7E
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr


if __name__ == '__main__':
    # 仿真环境
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])
    gm.gen_frame().attach_to(base)
    rbt_s = UR7E.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                   enable_cc=True)  # 仿真机器人
    obstacle_list = rbt_s.get_obstacle_list(base,False)

    rbt_s.hnd.jaw_to(0.0)  # 机器人手爪0张开
    rbt_s.gen_meshmodel().attach_to(base)  # 将机器人模型添加到场景
    # start_conf = rbt_s.get_jnt_values(manipulator_name)
    start_conf = np.array([0,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,-math.pi/2])  # 起始关节角度
    # print(start_conf)
    hand_name = "hnd"  # 手爪
    # object放置初始物体
    objcm_name = "u"  # 物体
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    stl_path = os.path.join(current_dir, "u.STL")
    # 然后使用这个路径
    obj = cm.CollisionModel(stl_path)  # 创建碰撞模型
    # 初始物体
    obj_pos = np.array([0.232+0.16, 0.32+0.16, 0.9])
    obj_rot = rm.rotmat_from_axangle([0, 1, 0], math.pi)
    obj.set_rgba([.9, .75, .35, 1])  # 颜色
    obj.set_pos(obj_pos)  # 开始位置
    obj.set_rotmat(obj_rot)  # 开始姿态
    obj.show_localframe()
    obj.attach_to(base)

    # # object_goal目标位置
    # obj_goal = cm.CollisionModel(stl_path)
    # obj_goal.set_rgba([1, 1, 1, 1])  # 颜色
    # obj_goal.set_pos(np.array([.4, .4, 0.05]))  # 目标位置
    # obj_goal.set_rotmat(np.eye(3))  # 目标姿态
    # obj_goal.attach_to(base)
    #
    # # base.run()

    gripper_s = dh.Dh50()  # 创建手爪实例
    # 从pickle文件中获取抓取姿态
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=current_dir,
                                           file_name='U1_dh50.pickle')  # 从pickle文件中加载预计算抓取姿态
    print(grasp_info_list)
    jaw_center_info_list = []
    # base.run()
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        jaw_center_pos = obj_pos + obj_rot @ jaw_center_pos
        jaw_center_rotmat = jaw_center_rotmat @ obj_rot
        gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        if gripper_s.is_collided(obstacle_list):
            # gripper_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(base)
            pass
        else:
            # gripper_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(base)
            jaw_center_info_list.append([jaw_width,jaw_center_pos,jaw_center_rotmat])
    conf_list=[]
    print(f"jaw_center_info_list:{jaw_center_info_list}")
    seed = np.array([-0.057731628145258965, -1.3630699998055795, 1.542364674321553, -1.7500909358126648, -1.5707964340644267, -1.6285276902039398])
    for jaw_center_info in jaw_center_info_list: # 不碰撞的手爪列表
        jaw_width,jaw_center_pos, jaw_center_rotmat = jaw_center_info
        conf = rbt_s.tracik(tgt_pos=jaw_center_pos,
                       tgt_rotmat=jaw_center_rotmat,
                       seed_jnt_values=seed,
                        solver_type = "Distance")
        if conf is None:
            print("ik求解失败")
            gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            gripper_s.gen_meshmodel(rgba=[1, 1, 0, .8]).attach_to(base) # 黄色手爪
        else:   # ik求解成功
            rbt_s.fk("arm", conf)
            # 碰撞检测
            if not rbt_s.is_collided(obstacle_list):    # 没碰撞
                rbt_s.gen_meshmodel(rgba=[0, 1, 0, .8]).attach_to(base)  # 将机器人模型添加到场景
                conf_list.append(conf)
            else:   # 碰撞
                rbt_s.gen_meshmodel(rgba=[1, 0, 0, .8]).attach_to(base)  # 将机器人模型添加到场景

        # print(jaw_width)
        # rbt_s.jaw_to(jaw_width)
        # test_conf = rbt_s.ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat)
        # rbt_s.fk('arm', test_conf)
        # rbt_s.gen_meshmodel().attach_to(base)
    base.run()

    # start_pos = obj.get_pos()  # 开始位置
    # start_rotmat = obj.get_rotmat()  # 开始姿态
    # # ===================================
    # start_homo = rm.homomat_from_posrot(start_pos, start_rotmat)  # 将物体的位置和旋转矩阵转换为一个齐次变换矩阵
    # jnts_list = []
    # for grasp_info in grasp_info_list:  # 抓取姿态提取
    #     jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info  #
    #     jnts = rbt_s.ik(component_name="arm", tgt_pos=rm.homomat_transform_points(start_homo, jaw_center_pos),
    #                     tgt_rotmat=start_rotmat.dot(jaw_center_rotmat))  # 机器人名，位置，角度，ik求解
    #     jnts_list.append(jnts)  # IK求解结果（关节角度或None）添加到列表
    #
    # # print("开始位置的IK解：",  jnts_list)
    #
    # rrtc_planner = rrtc.RRTConnect(rbt_s)  # 创建RRT-Connect路径规划器实例，用于后续的运动规划
    #
    # goal_pos = obj_goal.get_pos()  # 目标位置
    # goal_rotmat = obj.get_rotmat()  # 目标姿态
    #
    # # goal_jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    # # rbt_s.fk(component_name="arm", jnt_values=goal_jnt_values)
    # # rbt_s.gen_meshmodel().attach_to(base)
    # # 创建RRT-Connect规划器和抓取放置规划器
    # rrtc_s = rrtc.RRTConnect(rbt_s)
    # ppp_s = ppp.PickPlacePlanner(rbt_s)
    # # 计算物体的起始和目标齐次变换矩阵
    # obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    # obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)
    # # conf_list：机器人手臂的关节角度序列（每个时刻的姿势）
    # # jawwidth_list：手爪宽度序列（什么时候张开、什么时候闭合）
    # # objpose_list：物体的姿态序列（被抓取后随机器人移动的轨迹
    # conf_list, jawwidth_list, objpose_list = \
    #     ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
    #                                     objcm=obj,
    #                                     grasp_info_list=grasp_info_list,
    #                                     start_conf=start_conf,
    #                                     end_conf=None,
    #                                     goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
    #                                     approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
    #                                     approach_distance_list=[.1] * 2,
    #                                     depart_direction_list=[np.array([0, 0, 1]), np.array([0, 0, 1])],
    #                                     depart_distance_list=[.1] * 2)  # 生成完整的抓取-放置运动轨迹
    # robot_attached_list = []
    # object_attached_list = []
    # counter = [0]
    #
    #
    # def update(robot_s,
    #            object_box,
    #            robot_path,
    #            jawwidth_path,
    #            obj_path,
    #            robot_attached_list,
    #            object_attached_list,
    #            counter,
    #            task):
    #
    #     if robot_path is None:
    #         print("警告：路径生成失败，无法继续执行")
    #         return  # 或其他错误处理逻辑
    #     if counter[0] >= len(robot_path):  # 循环播放轨迹
    #         counter[0] = 0
    #     if len(robot_attached_list) != 0:  # 清除上一帧的模型
    #         for robot_attached in robot_attached_list:
    #             robot_attached.detach()
    #         for object_attached in object_attached_list:
    #             object_attached.detach()
    #         robot_attached_list.clear()
    #         object_attached_list.clear()
    #     # 更新当前帧的机器人姿势
    #     pose = robot_path[counter[0]]
    #     robot_s.fk(manipulator_name, pose)  # 正运动学求解
    #     robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])  # 设置手爪宽度
    #     robot_meshmodel = robot_s.gen_meshmodel()  # 生成并添加新的机器人模型
    #     robot_meshmodel.attach_to(base)
    #     robot_attached_list.append(robot_meshmodel)
    #     # 更新物体位置
    #     obj_pose = obj_path[counter[0]]
    #     objb_copy = object_box.copy()
    #     objb_copy.set_rgba([1, 0, 0, 1])  # 设置抓取后的物体颜色为红色
    #     objb_copy.set_homomat(obj_pose)
    #     objb_copy.attach_to(base)
    #     object_attached_list.append(objb_copy)
    #     counter[0] += 1  # 更新计数器
    #     return task.again  # 继续执行任务
    #
    #
    # # 设置定时任务，每0.01秒更新一帧
    # taskMgr.doMethodLater(0.01, update, "update",
    #                       extraArgs=[rbt_s,
    #                                  obj,
    #                                  conf_list,
    #                                  jawwidth_list,
    #                                  objpose_list,
    #                                  robot_attached_list,
    #                                  object_attached_list,
    #                                  counter],
    #                       appendTask=True)
    # base.run()  # 运行主循环
