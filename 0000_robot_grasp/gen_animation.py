#仿真动画制作

import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.nextage.nextage as nxt
import motion.probabilistic.rrt_connect as rrtc
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76
import robot_sim.robots.gofa5.gofa5 as gf5
# 1.只让nut运动
'''
base = wd.World()#仿真环境
nut = cm.CollisionModel('./stl_model/nut.stl') #nut模型
nut.set_rgba([0.5, 0.5, 0.5, 1])  # 设置颜色
# 定义 nut 的运动路径
position_list = np.linspace([0, 0, 0], [0.1, 0, 0], 50)
# 位置计数器,列表类型
counter: list = [0]
# 添加参考坐标系
gm.gen_frame(length=0.05).attach_to(base)

def update(obj: cm.CollisionModel, path_list, counter, task):
    # 到达目标点后，回到初始点
    if counter[0] >= len(path_list):
        counter[0] = 0
    # 获取当前位置
    current_pos = path_list[counter[0]]
    # 更新 nut 的位置
    obj.set_pos(current_pos)
    obj.attach_to(base)
    # 可选：在当前位置显示一个小球（不会消失），用于可视化路径
    # gm.gen_sphere(pos=current_pos, radius=0.005, rgba=[1, 0, 0, 1]).attach_to(base)
    counter[0] += 1
    return task.again

if __name__ == '__main__':
    # 初始显示 nut
    nut.attach_to(base)
    # 启动更新任务，每0.02秒更新nut的位置
    taskMgr.doMethodLater(0.02, update, "update",
                          extraArgs=[nut, position_list, counter],
                          appendTask=True)  
    base.run()#环境更新
'''

#2.只让机械臂运动

base = wd.World()#仿真环境
robot_s = gf5.GOFA5()#机器人
rbt_pos_list = np.linspace([0.713, 0.063, 0.18], [0.613, 0.063, 0.18], 50)#机械臂末端位置路径
rbt_path_list_forward = []#机械臂前进关节路径
for rbt_pos in rbt_pos_list:#机械臂末端位置路径
    rbt_jnt = robot_s.ik('arm', rbt_pos, rm.rotmat_from_axangle([1, 0, 0], np.pi))  # 逆运动学求解
    # 正运动学检测
    '''
    robot_s.fk('arm', rbt_jnt)
    robot_s.gen_meshmodel().attach_to(base)
    base.run()
    '''
    rbt_path_list_forward.append(rbt_jnt)   # 加入机械臂前进关节路径

rbt_path_list_backward = rbt_path_list_forward[::-1]    # 机械臂后退关节路径
rbt_path_list = rbt_path_list_forward + rbt_path_list_backward  # 机械臂总关节路径=前进+后退
counter: list = [0]     # 列表类型的counter
robot_mesh_list = []    # 存储机器人的网格模型

grasp_width_list = [0.076] * len(rbt_path_list)     # 夹爪宽度路径

def update(obj: cm.CollisionModel, path_list, counter, task):
    # 到目标点后，返回初始点
    if counter[0] >= len(path_list):
        counter[0] = 0
    # 检查机器人网格模型列表是否不为空，如果列表中有模型，将每个模型从场景中移除
    if len(robot_mesh_list) > 0:
        for mesh in robot_mesh_list:
            mesh.detach()   # 模型从场景中移除
    # 正运动学求解
    robot_s.fk('arm', rbt_path_list[counter[0]])
    # 手爪运动
    robot_s.jaw_to(hand_name='hnd', jawwidth=grasp_width_list[counter[0]])
    # 根据当前机器人状态生成网格模型
    robot_mesh = robot_s.gen_meshmodel()
    # 网格模型添加到仿真环境中显示
    robot_mesh.attach_to(base)
    # 将新生成的模型添加到机器人网格模型列表中
    robot_mesh_list.append(robot_mesh)
    # 计数+1
    counter[0] += 1
    return task.again

if __name__ == '__main__':
    taskMgr.doMethodLater(0.02, update, "update",
                          extraArgs=[robot_s, rbt_path_list, counter],
                          appendTask=True)
    base.run()

# 3.只动夹爪
# base = wd.World()   # 仿真环境
# robot_s = gf5.GOFA5()   # 机器人
# # 机械臂末端在特定位置
# # 尝试多次逆运动学求解
# rbt_position = [0.6, 0, 0.5]  # 位置
# target_rotation = rm.rotmat_from_axangle([1, 0, 0], np.pi)  # 角度
# max_attempts = 50
# joint_positions1 = None
#
# joint_positions1 = robot_s.get_jnt_values(component_name='arm')#当前位置
# print(joint_positions1)
#
# counter: list = [0]     # 列表类型的counter
# robot_mesh_list = []    # 存储机器人的网格模型
#
# grasp_width_list = np.linspace(0, 0.076, 50).tolist()   # 夹爪宽度路径
# grasp_width_list.extend(grasp_width_list[::-1])     # 夹爪宽度路径
#
# def update(obj: cm.CollisionModel, path_list, counter, task):
#     # 到目标点后，返回初始点
#     if counter[0] >= len(path_list):
#         counter[0] = 0
#     # 检查机器人网格模型列表是否不为空，如果列表中有模型，将每个模型从场景中移除
#     if len(robot_mesh_list) > 0:
#         for mesh in robot_mesh_list:
#             mesh.detach()   # 模型从场景中移除
#     # 正运动学求解
#     robot_s.fk('arm', joint_positions1)
#     # 手爪运动
#     robot_s.jaw_to(hand_name='hnd', jawwidth=grasp_width_list[counter[0]])
#     # 根据当前机器人状态生成网格模型
#     robot_mesh = robot_s.gen_meshmodel()
#     # 网格模型添加到仿真环境中显示
#     robot_mesh.attach_to(base)
#     # 将新生成的模型添加到机器人网格模型列表中
#     robot_mesh_list.append(robot_mesh)
#     # 计数+1
#     counter[0] += 1
#     return task.again
#
# if __name__ == '__main__':
#     taskMgr.doMethodLater(0.02, update, "update",
#                           extraArgs=[robot_s, grasp_width_list, counter],
#                           appendTask=True)
#     base.run()

#4.先动机械臂再动夹爪

# base = wd.World()#仿真环境
# robot_s = gf5.GOFA5()#机器人
# rbt_pos_list = np.linspace([0.713, 0.063, 0.18], [0.613, 0.063, 0.18], 50)#机械臂末端位置路径
# rbt_path_list_forward = []#机械臂前进关节路径
# for rbt_pos in rbt_pos_list:#机械臂末端位置路径
#     rbt_jnt = robot_s.ik('arm', rbt_pos, rm.rotmat_from_axangle([1, 0, 0], np.pi))  # 逆运动学求解
#     # 正运动学检测
#     '''
#     robot_s.fk('arm', rbt_jnt)
#     robot_s.gen_meshmodel().attach_to(base)
#     base.run()
#     '''
#     rbt_path_list_forward.append(rbt_jnt)   # 加入机械臂前进关节路径
#
# rbt_path_list_backward = rbt_path_list_forward[::-1]    # 机械臂后退关节路径
# rbt_path_list = rbt_path_list_forward + rbt_path_list_backward  # 机械臂总关节路径=前进+后退
# counter: list = [0]     # 列表类型的counter
# robot_mesh_list = []    # 存储机器人的网格模型
#
# grasp_width_list = [0.076] * len(rbt_path_list)     # 夹爪宽度路径
#
# def update(obj: cm.CollisionModel, path_list, counter, task):
#     # 到目标点后，返回初始点
#     if counter[0] >= len(path_list):
#         counter[0] = 0
#     # 检查机器人网格模型列表是否不为空，如果列表中有模型，将每个模型从场景中移除
#     if len(robot_mesh_list) > 0:
#         for mesh in robot_mesh_list:
#             mesh.detach()   # 模型从场景中移除
#     # 正运动学求解
#     robot_s.fk('arm', rbt_path_list[counter[0]])
#     # 手爪运动
#     robot_s.jaw_to(hand_name='hnd', jawwidth=grasp_width_list[counter[0]])
#     # 根据当前机器人状态生成网格模型
#     robot_mesh = robot_s.gen_meshmodel()
#     # 网格模型添加到仿真环境中显示
#     robot_mesh.attach_to(base)
#     # 将新生成的模型添加到机器人网格模型列表中
#     robot_mesh_list.append(robot_mesh)
#     # 计数+1
#     counter[0] += 1
#     return task.again
#
# if __name__ == '__main__':
#     taskMgr.doMethodLater(0.02, update, "update",
#                           extraArgs=[robot_s, rbt_path_list, counter],
#                           appendTask=True)
#     base.run()



base = wd.World()#仿真环境
robot_s = gf5.GOFA5()#机器人
rbt_pos_list = np.linspace([0.713, 0.063, 0.18], [0.613, 0.063, 0.18], 50)#机械臂末端位置路径
rbt_path_list_forward = []#机械臂前进关节路径
for rbt_pos in rbt_pos_list:#机械臂末端位置路径
    rbt_jnt = robot_s.ik('arm', rbt_pos, rm.rotmat_from_axangle([1, 0, 0], np.pi))#逆运动学求解
    # robot_s.fk('arm', rbt_jnt)
    # robot_s.gen_meshmodel().attach_to(base)
    # base.run()
    rbt_path_list_forward.append(rbt_jnt)#加入机械臂前进关节路径

rbt_path_list_backward = rbt_path_list_forward[::-1]#机械臂后退关节路径
rbt_path_list_forward.extend(rbt_path_list_backward)
rbt_path_list = rbt_path_list_forward
counter: list = [0] #列表类型的counter

nut = cm.CollisionModel('./stl_model/nut.stl') #nut模型
position_list = np.linspace([0, 0, 0], [0.1, 0, 0], 50)
gm.gen_frame(length=0.05).attach_to(base)
robot_mesh_list = []# 存储机器人的网格模型

grasp_width_list = np.linspace(0, 0.076, 50).tolist()
grasp_width_list.extend(grasp_width_list[::-1])

def update(obj: cm.CollisionModel, path_list, counter, task):
    if counter[0] >= len(path_list):
        counter[0] = 0
    if len(robot_mesh_list) > 0:
        for mesh in robot_mesh_list:
            mesh.detach()
    robot_s.fk('arm', rbt_path_list[counter[0]])
    robot_s.jaw_to(hand_name='hnd', jawwidth=grasp_width_list[counter[0]])
    robot_mesh = robot_s.gen_meshmodel()
    robot_mesh.attach_to(base)
    robot_mesh_list.append(robot_mesh)

    # gm.gen_sphere(path_list[counter[0]]).attach_to(base)
    # obj.set_pos(path_list[counter[0]])
    # obj.attach_to(base)
    counter[0] += 1
    return task.again

if __name__ == '__main__':
    taskMgr.doMethodLater(0.02, update, "update",
                          extraArgs=[robot_s, rbt_path_list, counter],
                          appendTask=True)
    base.run()




