import numpy as np
import basis.robot_math as rm
# 世界坐标系的原点在机器人底座中心
# 固定参数字典
CONFIG = {
    'gasket': { # 垫片
        'place_pos': np.array([0.325 - 0.0125 - 0.025 * 2, 0.725 - 13.5 * 0.025, 0.03]),
        'place_rot': rm.rotmat_from_axangle([0, 1, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)),
        'M2': {'grip_z': 0.010 }, # 实际抓取高度
        'M3': {'grip_z': 0.010 },
        'M5': {'grip_z': 0.010 },
    },
    'blot': {   # 螺栓
        # 放置位置
        'place_pos': np.array([0.25-0.006, 0.65, 0.105 - 0.039]),
        # 放置旋转矩阵
        'place_rot': rm.rotmat_from_axangle([0, 1, 0], np.pi).dot(
            rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)) @ rm.rotmat_from_axangle([0, 1, 0], np.pi / 6),
        'M2': {
            'grip_z': 0.007,  # 单个物体实际抓取高度
            'split_stack_z': 0.013,  # 堆叠物体实际分开高度
            'standing_z': 0.018  # 单个直立物体实际推倒高度
        },
        'M3': {
            'grip_z': 0.008,  # 单个物体实际抓取高度
            'split_stack_z': 0.015,  # 堆叠物体实际分开高度
            'standing_z': 0.020  # 单个直立物体实际推倒高度
        },
        'M5': {
            'grip_z': 0.008,  # 单个物体实际抓取高度
            'split_stack_z': 0.015,  # 堆叠物体实际分开高度
            'standing_z': 0.020  # 单个直立物体实际推倒高度
        },
    },
    'nut': {    # 螺母
        # 放置位置
        'place_pos': np.array([0.3+0.001, 0.725-0.001, 0.045]),
        # 放置旋转矩阵
        'place_rot': rm.rotmat_from_axangle([0, 1, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)),
        'M2':{
            'grip_z': 0.012,    # 实际抓取高度
            'split_stack_z': 0.015, # 实际高度
        },
        'M3': {
            'grip_z': 0.012,  # 实际抓取高度
            'split_stack_z': 0.015,  # 实际高度
        },
        'M5': {
            'grip_z': 0.012,  # 实际抓取高度
            'split_stack_z': 0.015,  # 实际高度
        },
    },
    'plate': {  # 海绵平台
        'pos': np.array([0.075, 0.55, -0.02]) + np.array([0.085, 0.115, 0]),    # 位置
        'center': np.array([-0.04, 0.55+0.085, 0.11]),   # 海绵平台中心
        'rot_z': np.pi / 2, # z轴旋转角
        'rgba': [0, 0, 1, 0.5]  # 颜色
    },
    # 通用参数
    'robot': {
        'camera_pos': np.array([-0.010, 0.570, 0.15]),  # 相机拍摄位置
        'default_rot': rm.rotmat_from_axangle([0, 1, 0], np.pi).dot(rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)),
        "approach_dis": 0.1 # 接近距离
    },
    'camera': {
        'type': 'd435',
        'save_directory': 'Data_Intel_Realsense_d435'
    },
    'yolo': {
        'conf': 0.35,
    },
    'box': {
        'box_pos': np.array([-0.35, 0.75, 0.045]),
        'place_pos': np.array([-0.05+0.02, 0.625+0.02, 0.08]),
        'app_dist': 0.1
    },
    # 二维码在世界坐标系的位置
    'aruco': {
        'marker_0_real_pos': np.array([0.075 - 0.23, 0.550, 0.013]),    # 二维码0中心在世界坐标系的位置
        'marker_x_dist': 0.23,
        'marker_y_dist': 0.17,
        'use_default': False # 是否使用自定义二维码位置
    },
    # 二维码中心在原图中的位置
    'default_aruco_px': {0: np.array([65.27295684814453, 429.18621826171875]),    # 左下
                         1: np.array([585.7576293945312, 426.7193908691406]), # 右下
                         2: np.array([59.78061294555664, 48.55867385864258]),    # 左上
                         3: np.array([578.971923828125, 41.39285659790039])}      # 右上
}
