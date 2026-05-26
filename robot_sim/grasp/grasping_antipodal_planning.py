# 这段代码实现了一个基于仿真环境的抓取规划和控制任务
# 通过加载物体模型、计算抓取点并控制夹爪进行抓取,展示了机器人在仿真中的抓取过程
# 同时,计算出的抓取信息被保存到文件中,生成抓取位姿列表,方便后续使用

import math
import numpy as np
import os
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
#import robot_sim.end_effectors.gripper.dh60.dh60 as dh60
# import robot_sim.end_effectors.gripper.ag145.ag145 as dh #使用AG145的夹爪
import robot_sim.end_effectors.gripper.frank_research3.frank_research3 as dh #使用AG145的夹爪

if __name__ == '__main__':
    # 初始化仿真环境：设置摄像头位置和观察目标点
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    # 获取当前代码文件路径，拼接物体模型路径（相对路径，避免硬编码）
    this_dir, this_filename = os.path.split(__file__) #当前目录this_dir，当前文件夹名this_filename
    objpath = os.path.join(this_dir, 'cube50.STL') #物体路径：当前文件夹下

    # 检查文件是否存在
    if not os.path.exists(objpath):
        print(f"错误: 找不到物体模型文件 {objpath}")
        exit(1)

    # 加载物体模型：创建物体的碰撞模型

    objcm_name = 'cube50'#物体名

    filename = "{}_Franka.pickle".format(objcm_name)#抓取位姿列表pickle的文件名为{}_AG145_6.pickle

    object_tube = cm.CollisionModel(objpath)
    object_tube.set_rgba([1, 0, 0, 1])
    object_tube.attach_to(base)

    # 初始化夹爪：创建AG145型号夹爪
    gripper_s = dh.frank_research3()

    # 根据物体模型和夹爪类型来规划抓取点
    grasp_info_list = gpa.plan_grasps(gripper_s,
                                      object_tube,# 待抓取物体（试管架）
                                      angle_between_contact_normals=math.radians(177),# 夹爪两接触点法线夹角（177度）
                                      openning_direction='loc_y',   # 夹爪张开方向（沿夹爪本地坐标系X轴）
                                     # rotation_interval=math.radians(60),
                                      max_samples=10,  # 最大采样次数（最多生成2种抓法）
                                      min_dist_between_sampled_contact_points=.003, # 采样接触点最小间距（3毫米）
                                      contact_offset=.003)# 接触偏移量（夹爪与物体接触时预留3毫米间隙）

    gpa.write_pickle_file('cube50', grasp_info_list, './', filename)

    # 可视化所有抓取姿态：遍历抓法列表，在仿真环境中显示夹爪
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info  # 解构抓取信息,提取每个抓取点的相关数据
        gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)  # 根据抓取点的姿态、夹爪宽度以及夹爪的旋转矩阵控制夹爪进行抓取
        gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)  # 生成一个绿色的夹爪模型,并将其附加到仿真环境中,用于显示抓取操作
        print(jaw_width)  # 抓取宽度
        base.run()
    base.run()
