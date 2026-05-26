import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.nextage.nextage as nxt
import robot_sim.robots.ur7e.ur7ewithoutmachine as ur7

# 机器人路径仿真

base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
gm.gen_frame().attach_to(base)
rbt_s = ur7.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                 enable_cc=True)  # 仿真机器人
# rbt_s.gen_meshmodel().attach_to(base)
# 2. 障碍物模型
obstacle_list = rbt_s.get_obstacle_list(base, True)

gm.gen_frame().attach_to(base)  # 世界坐标系
rbt_s.gen_meshmodel().attach_to(base)  # 机器人

path2 = []

path = path2
# path = path2
# rbt_path_list_backward = rbt_path_list_forward[::-1]    # 机械臂后退关节路径
# rbt_path_list = rbt_path_list_forward + rbt_path_list_backward  # 机械臂总关节路径=前进+后退
counter: list = [0]     # 列表类型的counter
robot_mesh_list = []    # 存储机器人的网格模型


def update(obj: cm.CollisionModel, path_list, counter, task):
    # 到目标点后，返回初始点
    if counter[0] >= len(path_list):
        counter[0] = 0
    # 检查机器人网格模型列表是否不为空，如果列表中有模型，将每个模型从场景中移除
    if len(robot_mesh_list) > 0:
        for mesh in robot_mesh_list:
            mesh.detach()   # 模型从场景中移除
    # 正运动学求解
    rbt_s.fk('arm', path[counter[0]])

    # 根据当前机器人状态生成网格模型
    robot_mesh = rbt_s.gen_meshmodel()
    # 网格模型添加到仿真环境中显示
    robot_mesh.attach_to(base)
    # 将新生成的模型添加到机器人网格模型列表中
    robot_mesh_list.append(robot_mesh)
    # 计数+1
    counter[0] += 1
    return task.again

if __name__ == '__main__':
    global taskMgr
    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbt_s, path, counter],
                          appendTask=True)
    base.run()