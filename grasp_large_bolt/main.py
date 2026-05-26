import os
# import cv2
import numpy as np
import math
import time
import threading
import queue
import traceback
import ScaleAwareICP_part as scale  # icp匹配
import visualization.panda.world as wd  # 仿真环境
import modeling.geometric_model as gm
import basis.robot_math as rm
# 只保留必要的导入，注释掉机器人相关代码以便调试
import robot_sim.robots.ur7e.ur7e_withouttable as rbt
import robot_con.ur.ur7_dh50_rtde as ur7con
import open3d as o3d
import Mech_camera as Mech
from config import CONFIG
import modeling.collision_model as cm
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd
import robot_planning as robotplanner
import grasp_planner as gp

if __name__ == "__main__":

    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    rbt_s = rbt.UR7E(enable_cc=True)  # 仿真机器人
    rbt_s.gen_meshmodel().attach_to(base)
    obstacle_list = rbt_s.get_obstacle_list(base, False)
    gripper_s = hnd.Dh50()
    rbt_r = ur7con.UR5Ag95X_RTDE(robot_ip="192.168.125.30")

    pickplace = robotplanner.RobotPlanner(rbt_s, rbt_r, None, gripper_s, CONFIG, base, sim=False)
    grasp_planner = gp.GraspPlanner(obj_name='M16_50', gripper_sim=gripper_s)
    pickplace.init_rbt()
    # conf = np.array([1.1737035512924194, -1.3214591157487412, 2.1930721441852015, -2.447783132592672, -1.5962050596820276, 1.242830514907837])
    # rbt_s.fk('arm',conf)
    # rbt_s.hnd.close()
    # rbt_s.gen_meshmodel().attach_to(base)
    # base.run()
    icp = scale.RotationOnlyICP()  # icp匹配
    mech = Mech.CaptureImage()

    RGB_image, Depth_image, ply_file = mech.capture_and_generate_pointcloud(
        save=True,pcb_out_path=r"E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\images\Mech\pointcloud.ply",
        depth_trunc=3,keep_invalid=True, show=False
    )
    # pcd_path = r"E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\images\Mech\pointcloud.ply"
    # pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_np = np.asarray(ply_file.points)    # 将点云转化为np数据
    # 点云坐标变换
    T_cam_to_world = CONFIG['T_cam_to_world']   # 相机外参
    pcd = mech.transform_point_cloud(pcd_np, T_cam_to_world)
    # gm.gen_pointcloud(pcd,rgbas = [[0, 1, 0, .3]]).attach_to(base)  # 把变换后的点云放到仿真环境里

    # # 找支架位置
    # # #取一定范围内点云
    # M16table_filter_pcd = mech.filter_pointcloud_by_range(pcd, x_range=(0.5,0.75), y_range=(-0.2,-0.1), z_range=(0.03,0.5))
    # gm.gen_pointcloud(M16table_filter_pcd, rgbas=[[1, 1, 0, .3]]).attach_to(base)  # 把变换后的点云放到仿真环境里
    # # 转换为 Open3D 点云
    # M16table_filtered_pcd_obj = o3d.geometry.PointCloud()
    # M16table_filtered_pcd_obj.points = o3d.utility.Vector3dVector(M16table_filter_pcd)
    # # icp匹配
    # M16table_transformation = icp.simple_icp_with_scale_fix(
    #     M16table_filtered_pcd_obj, ply_path = None, stl_path =r'E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\object\M16table_surface.STL',
    #     manual_scale=True, scene_down_num = 0.001, use_cluster_center=True, use_bbox_center=True, eps=0.02,
    #     icp_iteration = 200, min_samples=30, max_distance = 0.01, show = False
    # )
    # print(f"架子transformation:{M16table_transformation}")
    # # icp匹配后的位姿
    # M16table_pos = M16table_transformation[:3,3]
    # M16table_rot = M16table_transformation[:3,:3]
    # M16table2 = cm.CollisionModel(
    #     r"E:\py_project\wrsrobot\wrs_shu\robot_sim\robots\ur7e\meshes\M16table.STL",
    #     cdprimit_type="box", expand_radius=0.001  # 碰撞体扩大
    # )
    # M16table2.set_pos(M16table_pos)
    # M16table2.set_rotmat(M16table_rot)
    # M16table2.set_rgba(rgba = [1, 0, 0, 0.5])
    # M16table2.attach_to(base)

    # 找放置孔位，取一定范围内点云
    M16hole_filter_pcd = mech.filter_pointcloud_by_range(pcd, x_range=(0.5, 0.75), y_range=(-0.05, 0.1),
                                                          z_range=(0.045, 0.08))
    gm.gen_pointcloud(M16hole_filter_pcd, rgbas=[[1, 1, 0, .3]]).attach_to(base)  # 把变换后的点云放到仿真环境里
    # 转换为 Open3D 点云
    M16hole_filtered_pcd_obj = o3d.geometry.PointCloud()
    M16hole_filtered_pcd_obj.points = o3d.utility.Vector3dVector(M16hole_filter_pcd)
    M16hole_transformation = icp.simple_icp_with_scale_fix(
        M16hole_filtered_pcd_obj, ply_path=None,
        stl_path=r'E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\object\M16hole_surface.STL',
        manual_scale=True, scene_down_num = 0.001, use_cluster_center=True, use_bbox_center=True, eps=0.01,
        icp_iteration = 200, min_samples=30, max_distance=0.005, show=False
    )
    print(f"放置位置transformation:{M16hole_transformation}")
    # icp匹配后的位姿
    M16hole_pos = M16hole_transformation[:3, 3]
    M16hole_rot = M16hole_transformation[:3, :3]
    M16hole2 = cm.CollisionModel(
        r"E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\object\M16hole.STL",
        cdprimit_type="box", expand_radius=0.001  # 碰撞体扩大
    )
    M16hole2.set_pos(M16hole_pos)
    M16hole2.set_rotmat(M16hole_rot)
    M16hole2.set_rgba(rgba=[1, 0, 0, 0.5])
    M16hole2.attach_to(base)

    place_pos = M16hole_pos+np.array([0.002,0.002,0.07])
    print(f"放置位置：{place_pos}")
    place_transformation = M16hole_transformation.copy()
    # 修改旋转部分：右乘绕 Y 轴 180° 的旋转矩阵
    place_transformation[:3, :3] = M16hole_transformation[:3, :3] @ rm.rotmat_from_axangle([0, 1, 0], math.pi)
    place_transformation[:3, 3] = place_pos
    print(f"放置齐次变换矩阵：{place_transformation}")

    # 找螺栓位置，取一定范围内点云
    M16bolt_filter_pcd = mech.filter_pointcloud_by_range(pcd, x_range=(0.5,0.75), y_range=(-0.2,-0.1), z_range=(0.058,0.065))
    gm.gen_pointcloud(M16bolt_filter_pcd, rgbas=[[1, 1, 0, .3]]).attach_to(base)  # 把裁剪后的点云放到仿真环境里
    # # 转换为 Open3D 点云
    M16bolt_filtered_pcd_obj = o3d.geometry.PointCloud()
    M16bolt_filtered_pcd_obj.points = o3d.utility.Vector3dVector(M16bolt_filter_pcd)
    # icp匹配
    M16bolt_transformation = icp.simple_icp_with_scale_fix(
        M16bolt_filtered_pcd_obj, ply_path=None,
        stl_path=r'E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\object\M16_surface.STL',
        manual_scale=True, scene_down_num = 0.00001, use_cluster_center=True, use_bbox_center=True, eps=0.01,
        icp_iteration=1000, min_samples=30, max_distance=0.0005, show=False
    )
    # 复制变换矩阵（确保可写）
    M16bolt_transformation = M16bolt_transformation.copy()
    # 修改旋转部分：右乘绕 Y 轴 180° 的旋转矩阵
    M16bolt_transformation[:3, :3] = M16bolt_transformation[:3, :3] @ rm.rotmat_from_axangle([0, 1, 0], math.pi)
    print(f"螺栓位置transformation:{M16bolt_transformation}")
    # icp匹配后的位姿
    M16bolt_pos = M16bolt_transformation[:3, 3]
    M16bolt_rot = M16bolt_transformation[:3, :3]

    M16bolt2 = cm.CollisionModel(
        r"E:\py_project\wrsrobot\wrs_shu\grasp_large_bolt\object\M16_50.STL",
        cdprimit_type="box", expand_radius=0.001  # 碰撞体扩大
    )
    M16bolt2.set_pos(M16bolt_pos)
    M16bolt2.set_rotmat(M16bolt_rot)
    M16bolt2.set_rgba(rgba=[1, 0, 0, 0.5])
    M16bolt2.attach_to(base)
    # base.run()
    object_trans_mat = M16bolt_transformation
    target_trans_mat = place_transformation

    pick_path, place_path, grasp_info = pickplace.plan_pick_place(object_trans_mat, target_trans_mat, grasp_planner)
    if pick_path is None or place_path is None:
        print("No valid path found.")
    else:
        pickplace.execute_pick_place(target_trans_mat, pick_path, place_path)


    base.run()





