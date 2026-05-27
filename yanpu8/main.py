import os
# import cv2
import numpy as np
import math
import time
import threading
import queue
import traceback
import yolo_Mask as YOLO # yolo检测
import Mask_clipping_point_cloud as mcpc    # 点云分割
import ScaleAwareICP_part as scale  # icp匹配
import visualization.panda.world as wd  # 仿真环境
import modeling.geometric_model as gm
import basis.robot_math as rm
# 只保留必要的导入，注释掉机器人相关代码以便调试
import robot_sim.robots.ur7e.ur7ewithoutmachine as cbt
import robot_con.ur.ur7_dh50_rtde as ur7con
import path_plan as pp  # 路径规划
import Mech_camera as Mech  # 梅卡曼德
import D435camera as D435   # D435
import drivers.devices.dh.ag145 as gripper  # dh手爪
import force_control_move as fc # 力控
import socket_get_image as sgi  # 双进程获取图像
import vacuum_clean as vc   # 吸盘程序
from config import CONFIG_U1,CONFIG_U625    # 固定参数
import sys
# 抓取U1倒放和U625程序,外撑式手爪
# 放置路径固定，只计算抓取路径

def clear_queue(q: queue.Queue):   # 清除队列
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass

def robot_simulation(path,rgba):
    rbt_mesh = None
    for jnts1 in path:
        rbt_s.fk("arm", jnts1)  ## 用正运动学（fk）更新机器人姿态
        rbt_mesh = rbt_s.gen_meshmodel(rgba=rgba)
        rbt_mesh.attach_to(base)
    return rbt_mesh

def move_to_point(robot_real, path_plan, target_conf, fast_v, fast_a, slow_v, slow_a):
    """
    从当前位置移动到目标位置，用于回到wait_conf
    """
    # 检查robot_real是否掉线
    if robot_real.check_rtder_is_connected():
        print("未掉线")
    else:
        print("rtde_r重连失败,程序终止")
        sys.exit(1)  # 程序终止
    # 获取当前关节点
    current_conf_0 = robot_real.get_jnt_values()  # 当前关节点
    # 从当前位置慢速移动到target_conf
    if (current_conf_0 != target_conf).any():  # 移动到等待位置
        start_path_0 = path_plan.plan_path(current_conf_0, target_conf)
        robot_real.move_jnts(start_path_0[0], slow_v, slow_a)  # 到初始点，保险起见
        robot_real.move_jntspace_path(
            path=start_path_0,
            interval_time=1,
            control_frequency=.002,
            vel=fast_v, acc=fast_a,
            speed_gain=300,
            blend=0.0,
            toppra_vels=[fast_v] * 6,
            toppra_accs=[fast_a] * 6,
        )
    robot_real.move_jnts(target_conf, slow_v, slow_a)  # 保险起见
    print("已到达target_conf")

"""处理梅卡曼德远处拍摄的图像和点云数据，返回是否检测到掩码，对齐后的变换矩阵"""
def process_image_and_pointcloud(color_image_0, pcb_image_0, grasp_success_0, mask_center_0):
    """
    处理实时采集的图像和点云数据
    Args:
        color_image_0: 实时采集的彩色图像
        pcb_image_0: 实时合成的点云图
        grasp_success_0: 上次抓取是否成功
        mask_center_0: 上次抓取的掩码中心
    Returns:
        has_mask_1: 是否有有效的掩码
        cls_id_1: 物体正反种类
        pos_1: 抓取位置
        mask_center_1: 这次抓取的掩码中心
        angle_1: 物体的角度
    """
    try:
        # 0.图片中箱子外面涂黑
        # 创建全黑背景图（与原图尺寸相同）
        box_image = np.zeros_like(color_image_0)  # 全黑
        # 将箱子框内区域复制到黑图上，中心坐标（宽1020，高500）
        box_image[50:950, 340:1700] = color_image_0[50:950, 340:1700]
        # 1.yolo实例分割，输出掩码的路径
        print("yolo实例分割")
        cls_id_1, mask, mask_center_1, cropped_image  = Mech_seg_model.seg_objs_by_yolo_2(
            box_image, mask_center_0, grasp_success_0,
            show=False, save=True,
        )
        if cls_id_1 is None:
            print("未检测到物体")
            return False, None, None, None, None
        # 检测到物体
        has_mask_1 = True
        # 2. 点云裁剪
        print("裁剪点云...")
        cropped_pcd = mask_cropper.main(pcb_image_0, mask, color_image_0, Mech_ply_path, show=False)
        # print(f"裁剪后点云保存到: {ply_path}")
        # 3.求掩码矩形内物体的旋转角度
        if cls_id_1 == 1: # 检测到正面
            angle_1 = 0
            # 3. ICP配准
            print("进行ICP配准...")
            # 使用包围盒中心对齐，平移且旋转，输出平移矩阵
            point_cam = icp.simple_icp_with_scale_fix(
                ply=cropped_pcd, ply_path=Mech_ply_path, stl_path=STL1_PATH, manual_scale=1,
                use_cluster_center=True, use_bbox_center=True,
                eps=0.02, min_samples=30,
                max_distance=0.02,
                show=False,   # 是否展示
            )
        else:   # 检测到反面
            rotated_rect = Mech_seg_model.get_rotated_rect_from_mask(mask) # 掩码的最小外接矩形
            angle_1 = far_angle_model.detect_angle_by_yolo(cropped_image, rotated_rect, show = False)
            if angle_1 is None:
                angle_1 = 0
            # ICP配准
            print("进行ICP配准...")
            # 使用包围盒中心对齐，平移且旋转，输出平移矩阵
            point_cam = icp.simple_icp_with_scale_fix(
                ply=cropped_pcd, ply_path=Mech_ply_path, stl_path=STL_PATH, manual_scale=1,
                use_cluster_center=True, use_bbox_center=True,
                eps=0.02, min_samples=30,
                max_distance=0.02,
                show=False,  # 是否展示
            )
        # print(f"point_cam: {point_cam}")
        # 外参矩阵
        T_cam_to_world = CONFIG_U1['T_cam_to_world']
        # 将相机坐标系转化为世界坐标系
        pos_1 = mech.cam_to_world(point_cam, T_cam_to_world)
        # point_cam = np.asarray(point_cam).reshape(1, 3)  # 确保是二维
        # # print(f"point_cam: {point_cam}")
        # # 转换为齐次坐标 (N, 4)
        # points_hom = np.hstack([point_cam, np.ones((1, 1))])
        # points_world_hom = (T_cam_to_world @ points_hom.T).T  # 得到 (N, 4),.T表示转置
        # pos_1 = points_world_hom[0, :3]

        pos_1 += np.array([-0.002, -0.0015, 0])    # 位置补偿
        # print(f"抓取位置矩阵pos_1: {pos_1}")
        if cls_id_1 == 1: # 检测到正面,比反面抓取更低
            pos_1 += np.array([0, 0, -0.005])
        return has_mask_1, cls_id_1, pos_1, mask_center_1, angle_1
    except Exception as b:
        print(f"✗ 图像和点云处理失败: {b}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None

def run_1_camera():
    """
    运行梅卡曼德第一次拍摄，并将结果放入 result_queue。
    返回 True 表示成功（已放入结果元组），False 表示失败（已放入 None）。
    """
    # 1. 采集初始数据,单进程
    color_image1, depth_image1, pcb_image1 = mech.capture_and_generate_pointcloud(
        save=True, show=False, pcb_out_path=Mech_pcb_out_path
    )
    if pcb_image1 is None:
        return False
    # 2. 处理数据
    has_mask1, cls_id_1, pos_1, mask_center2, angle_1 = process_image_and_pointcloud(
        color_image1, pcb_image1, grasp_success_0=True, mask_center_0=CONFIG_U1['grasp']['mask_center'],
    )
    if not has_mask1:
        print("无掩码")
        return False
    if angle_1 is None:
        angle_1 = 0 # 物体角度
    # 3. 计算路径
    # U1位置
    U1_pos_1 = pos_1 + np.array([0, 0, U1_pick_height]) # 加上补偿
    print(f'U1物体在世界坐标系下位置pos_1:{U1_pos_1}')
    # 抓取路径
    U1_grasp_and_place_path_1 = Path_plan.U1_grasp_and_place_path(cls_id_1, U1_pos_1, angle_1, wait_rot)
    if U1_grasp_and_place_path_1 is None:
        clear_queue(Mech_queue)
        Mech_queue.put(None)
        print("抓取路径计算失败")
        return False
    # 成功，放入结果元组
    grasp_tuple_1 = (cls_id_1, U1_grasp_and_place_path_1)
    clear_queue(Mech_queue)  # 确保队列为空
    Mech_queue.put(grasp_tuple_1)  # 抓取结果
    return True

"""机器人线程函数,先抓U1，再抓U625"""
def thread_robot(loop_count_0):
    print(f"[机器人线程] 第 {loop_count_0} 次循环开始")  # 此时在放置点上方
    global U1_place_num, U625_place_num, U1_pick_height,  U1_grasp_success, U625_grasp_success
    global U1_need_vacuum_clean, U625_need_vacuum_clean, current_point
    global End, U1_End, U625_End  # 结束标志
    fast_v = 2
    slow_v = 0.2
    fast_a = 2
    slow_a = 0.2
    try:
        if End:
            prepare_Mech_camera.set()  # 相机线程避免一直等待
            prepare_D435_camera.set()  # 相机线程避免一直等待
            return
        if U1_End or not U625_grasp_success:  # U1抓取结束，回到等待点。或者上次U625抓取失败，则重新抓取U625而不抓U1
            print("U1抓取结束，回到等待点")
            rbt_r.moveL(CONFIG_U1['grasp']['wait_pose'], fast_v, fast_a)  # 回到等待点
            prepare_D435_camera.set()  # D435_1相机开始拍摄
            current_point = 'wait_conf'
        else:   # 抓取U1
            # 全都从队列获取新数据
            print("[机器人线程] 等待梅卡曼德相机计算抓取路径结果")
            try:
                Mech_grasp_result = Mech_queue.get(timeout=30.0)
            except queue.Empty:
                print("[机器人线程] 等待梅卡曼德相机结果超时，还可以抓取U625")
                Mech_grasp_result = None
                U1_End = True   # U1抓取结束
                print("U1抓取结束，回到等待点")
                rbt_r.moveL(CONFIG_U1['grasp']['wait_pose'], fast_v, fast_a)  # 回到等待点
                current_point = 'wait_conf'

            if not U1_End:   # 梅卡曼德相机线程结果获取成功
                # 是否执行吸盘
                # U1_need_vacuum_clean = False    # 暂定
                if U1_need_vacuum_clean:
                    # 回到等待点
                    rbt_r.moveL(CONFIG_U1['grasp']['wait_pose'], fast_v, fast_a)  # 回到等待点
                    prepare_D435_camera.set()  # D435_1相机拍摄
                    print("开始U1吸盘操作")
                    # 手爪打开
                    gripper_target.put((100,100))  # 异步，不等待
                    gripper_ready_event.wait()
                    gripper_ready_event.clear()
                    # 移到抓取点
                    VC.vacuum_clean_by_layer_U1_1(U1_layer_num, 0.5, 0.5, 0.1, 0.1)
                    # 抓取
                    gripper_target.put(0)  # 异步，不等待
                    gripper_ready_event.wait()
                    gripper_ready_event.clear()
                    # 完成吸盘操作，回到放置位置
                    VC.vacuum_clean_by_layer_U1_2(U1_layer_num, 0.2, 0.2, 0.1, 0.1)
                    # 放手
                    gripper_target.put(100)  # 异步，不等待
                    gripper_ready_event.wait()
                    gripper_ready_event.clear()
                    # 回到等待点
                    VC.vacuum_clean_by_layer_U1_3(U1_layer_num, 0.5, 0.5, 0.1, 0.1)
                    # 执行吸盘操作
                    current_point = 'wait_conf'  # 记录当前点
                else:   # 不吸盘
                    # 如果队列里是None，表示没有检测到掩码
                    if Mech_grasp_result is None:
                        print("[机器人线程] 获取到梅卡曼德线程计算结果为空")
                        U1_End = True
                        # 回到等待位置
                        move_to_point(rbt_r, Path_plan, wait_conf, fast_v, fast_a, slow_v, slow_a)
                        current_point = 'wait_conf'  # 记录当前点
                        prepare_D435_camera.set()  # D435_1相机拍摄
                    else:   # 有结果
                        # 解包结果，前一循环的检测结果(局部变量)
                        (U1_cls_id, U1_grasp_and_place_path) = Mech_grasp_result
                        print("[机器人线程] 获取到新的路径数据")
                        # 执行机器人抓取
                        print("执行机器人抓取物体U1")
                        # 手爪闭合
                        gripper_target.put(0)  # 异步，不等待
                        print("手爪闭合")
                        # 执行准备路径,到达抓取点上方高处
                        if current_point == 'wait_conf':  # 当前在等待点
                            # 接近路径，快速，从等待点到抓取低点
                            U1_approach_path = U1_grasp_and_place_path["app"]
                        else:   # 当前在U625的放置高点
                            # 合并U625放置返回路径和U1抓取路径,平滑处理,用时0.02s左右
                            U1_approach_path = Path_plan.smooth_two_path(CONFIG_U625['place']['back_path'], U1_grasp_and_place_path["app"], 0.15)
                        # 执行接近路径
                        rbt_r.move_jnts(U1_approach_path[0],slow_v,slow_a) # 保险起见
                        rbt_r.move_jntspace_path(
                            path=U1_approach_path,
                            interval_time=1,
                            control_frequency=.002,
                            vel=fast_v, acc=fast_a,
                            speed_gain=300,
                            blend=0.0,
                            toppra_vels=[fast_v] * 6,
                            toppra_accs=[fast_a] * 6
                        )
                        ######
                        prepare_D435_camera.set()  # D435_1相机开始拍摄
                        ######
                        # 获取抓取低点位姿，改成从仿真中计算
                        low_grasp_pose = U1_grasp_and_place_path["low_grasp_pose"]
                        print(f"最终抓取低点位姿:{low_grasp_pose}")
                        # 抓取点位姿：改成从仿真中计算
                        grasp_pose = U1_grasp_and_place_path["grasp_pose"]
                        # 等待手爪运动完成
                        gripper_ready_event.wait()
                        gripper_ready_event.clear()
                        # 机器人下移到抓取点(高度与物体刚好接触)，慢速
                        rbt_r.moveL(grasp_pose,slow_v,slow_a)
                        # time.sleep(20)
                        # 力控下移抓取
                        ForceControl.move(time1 = 0.3, timeout=1, force = -10)
                        # 抓取
                        print("手爪抓取")
                        gripper_target.put((80,50))  # 异步，不等待
                        gripper_ready_event.wait()
                        gripper_ready_event.clear()
                        # 检测是否抓取成功，如果抓取失败，再下移抓取一遍
                        gripper_target.put('width')
                        gripper_ready_event.wait()
                        gripper_ready_event.clear()
                        width = gripper_current_width   # 手爪当前宽度
                        if U1_cls_id == 0:  # 倒放物体
                            print(f"U1倒放物体，手爪宽度:{width}")
                            U1_grasp_success = True if 10 < width < 70 else False  # 抓住时在36左右
                        else:  # 正放物体
                            print(f"U1正放物体，手爪宽度:{width}")
                            U1_grasp_success = True if 40 < width < 70 else False  # 抓住时在50左右
                        if not U1_grasp_success:    # 抓取失败
                            # gripper_r.close_g()  # 手爪闭合
                            # 上移2mm
                            rbt_r.moveL(grasp_pose+np.array([0,0,0.002,0,0,0]), slow_v, slow_a)
                            # 闭合
                            gripper_target.put(0)  # 异步，不等待
                            # 同时下移回抓取点
                            rbt_r.moveL(grasp_pose, slow_v, slow_a)
                            gripper_ready_event.wait()
                            gripper_ready_event.clear()
                            print("手爪闭合")
                            # 力控下移抓取
                            ForceControl.move(time1 = 0.5, timeout=1.2, force = -10)
                            # 第二次抓取
                            print("手爪第二次抓取U1")
                            gripper_target.put(80)  # 异步，不等待
                            gripper_ready_event.wait()
                            gripper_ready_event.clear()
                        # 上升路径
                        print("moveL到抓取低位")
                        rbt_r.moveL(low_grasp_pose-np.array([0,0,0.001,0,0,0]), slow_v, slow_a)    # 上移到抓取低点下面3mm处，避免运动到高于实际抓取低点的点
                        # 检测是否抓取成功，必须在梅卡曼德相机线程计算前
                        gripper_target.put(80)  # 异步，不等待
                        gripper_ready_event.wait()
                        gripper_ready_event.clear()
                        gripper_target.put('width') # 获取宽度
                        gripper_ready_event.wait()
                        gripper_ready_event.clear()
                        width = gripper_current_width  # 手爪当前宽度
                        if U1_cls_id == 0: # 倒放物体
                            print(f"U1倒放物体，手爪宽度:{width}")
                            U1_grasp_success = True if 10<width<70 else False  # 抓住时在36左右
                        else:   # 正放物体
                            print(f"U1正放物体，手爪宽度:{width}")
                            U1_grasp_success = True if 40<width<70 else False  # 抓住时在50左右
                        if not U1_grasp_success:    # 没有抓取成功
                            # 回到等待点
                            print("没抓住U1，回到等待点")
                            # 从U1的抓取低点回到等待点
                            U1_low_grasp_to_wait_path = U1_grasp_and_place_path["app"][::-1]
                            rbt_r.move_jnts(U1_low_grasp_to_wait_path[0],slow_v,slow_a)    # 必须有
                            rbt_r.move_jntspace_path(
                                path=U1_low_grasp_to_wait_path,  # 接近路径的反路径
                                interval_time=1,
                                control_frequency=.002,
                                vel=fast_v, acc=fast_a,
                                speed_gain=300,
                                blend=0.0,
                                toppra_vels=[fast_v] * 6,
                                toppra_accs=[fast_a] * 6
                            )
                            current_point = 'wait_conf' # 记录当前点
                            prepare_Mech_camera.set()  # 梅卡曼德相机开始拍摄
                            return  # 需要重新抓U1，不去抓U625
                        else:   # 抓取成功
                            if U1_cls_id == 1:  # 正放物体，抛弃
                                print("抓住了U1正放，丢弃")
                                # 从U1的抓取低点回到抛弃点，丢弃
                                rbt_r.move_jnts(U1_grasp_and_place_path["place"][0], slow_v, slow_a)  # 必须有
                                rbt_r.move_jntspace_path(
                                    path=U1_grasp_and_place_path["place"],  # 也计算了抛弃路径
                                    interval_time=1,
                                    control_frequency=.002,
                                    vel=fast_v, acc=fast_a,
                                    speed_gain=300,
                                    blend=0.0,
                                    toppra_vels=[fast_v] * 6,
                                    toppra_accs=[fast_a] * 6
                                )
                                # 松手
                                for _ in range(2):
                                    gripper_target.put(0)  # 异步，不等待
                                    gripper_ready_event.wait()  # 等待手爪动作完成
                                    gripper_ready_event.clear()  # 复位
                                    gripper_target.put('width')
                                    gripper_ready_event.wait()  # 等待完成
                                    gripper_ready_event.clear()  # 复位
                                    width = gripper_current_width
                                    if width <= 5:
                                        break
                                    gripper_target.put(100)  # 异步，不等待
                                    gripper_ready_event.wait()  # 等待手爪动作完成
                                    gripper_ready_event.clear()  # 复位
                                    gripper_target.put('width')
                                    gripper_ready_event.wait()  # 等待完成
                                    gripper_ready_event.clear()  # 复位
                                    width = gripper_current_width
                                    if width >= 95:
                                        break
                                else:
                                    print("丢弃失败,程序异常，结束")
                                    End = True
                                    prepare_Mech_camera.set()
                                    prepare_D435_camera.set()
                                    return
                                # 回到等待点
                                rbt_r.move_jnts(CONFIG_U1['abandon']['back_path'][0], slow_v, slow_a)  # 必须有
                                rbt_r.move_jntspace_path(
                                    path=CONFIG_U1['abandon']['back_path'],  #
                                    interval_time=1,
                                    control_frequency=.002,
                                    vel=fast_v, acc=fast_a,
                                    speed_gain=300,
                                    blend=0.0,
                                    toppra_vels=[fast_v] * 6,
                                    toppra_accs=[fast_a] * 6
                                )
                                current_point = 'wait_conf' # 记录当前点

                            else:   # 抓住了反面物体，离开并放置
                                print("执行机器人放置物体U1")
                                rbt_r.move_jnts(U1_grasp_and_place_path["place"][0], slow_v, slow_a)  # 必须有
                                # 检测放置区域是否有物体
                                camera_detect_time = time.time()
                                while True: # 一直检测到无物体
                                    # 相机拍摄RGB图
                                    rgb_image = D435_2.capture_rgb(delay = 0.1, show=False, save=False)
                                    if not U1_detect_model.detect_exist_U1(
                                            rgb_image,x_range=(130, 270),y_range=(300, 450),save=False,show = False):
                                        print("检测到放置区域无U1，可以放置")
                                        break
                                    if time.time() - camera_detect_time > 30:
                                        End = False
                                        print("等待物体放置超时，程序结束")
                                        return
                                    time.sleep(1)
                                    print("U625放置区域有物体")
                                print(f"检测U1放置区域用时{time.time()-camera_detect_time}")

                                rbt_r.move_jntspace_path(
                                    path=U1_grasp_and_place_path["place"],
                                    interval_time=1,
                                    control_frequency=.002,
                                    vel=fast_v, acc=fast_a,
                                    speed_gain=300,
                                    blend=0.0,
                                    toppra_vels=[fast_v] * 6,
                                    toppra_accs=[fast_a] * 6,
                                )   # 从抓取点移到放置高点，快速
                                # 机器人下移到放置点，慢速
                                rbt_r.moveL(U1_grasp_and_place_path["place_pose"], slow_v, slow_a)
                                gripper_target.put(10)  # 异步，不等待
                                gripper_ready_event.wait()  # 等待手爪动作完成
                                gripper_ready_event.clear()  # 复位
                                print("手爪放开")
                                U1_place_num += 1
                                print(f"已放置{U1_place_num}个物体U1")
                                # 回到放置高点，慢速
                                rbt_r.moveL(U1_grasp_and_place_path["high_place_pose"], slow_v, slow_a)
                                # 返回等待点,起点姿态不固定，所以用moveL。为了梅卡曼德能第一时间拍照，不将返回路径和抓取路径合并
                                rbt_r.moveL(CONFIG_U1['grasp']['wait_pose'], fast_v, fast_a)
                                current_point = 'wait_conf'
        #####################开始抓取U625物体########################
        if U625_End:  # U625抓取结束，回到等待点
            print("U625抓取结束，回到等待点")
            rbt_r.moveL(wait_pose, fast_v, fast_a)  # 回到等待点
            current_point = 'wait_conf'
            prepare_Mech_camera.set()  # 梅卡曼德相机开始拍摄
        else:
            # 闭合手爪
            gripper_target.put(0)  # 异步，不等待
            # 获取D435线程计算结果
            try:
                D435_result = D435_queue.get(timeout=30.0)
            except queue.Empty:
                print(f"等待相机处理结果超时")
                D435_result = None
                U625_End = True
                # 回到等待点
                rbt_r.moveL(wait_pose, fast_v, fast_a)  # 回到等待点
                current_point = 'wait_conf'
                gripper_ready_event.wait()  # 等待手爪动作完成
                gripper_ready_event.clear() # 复位

            if not U625_End:    # D435相机结果收到
                print("梅卡曼德相机线程开始采集")
                prepare_Mech_camera.set()
                # 是否吸盘
                # U625_need_vacuum_clean = False  # 暂定
                if U625_need_vacuum_clean:
                    # 回到等待点
                    rbt_r.moveL(wait_pose, fast_v, fast_a)  # 回到等待点
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    # path_down = D435_result
                    print("开始U625吸盘操作")
                    print(f"U625层数：{U625_layer_num}")
                    # 手爪打开
                    gripper_target.put((100,100))  # 异步，不等待
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    # 移到抓取点
                    VC.vacuum_clean_by_layer_U625_1(U625_layer_num, 0.5, 0.5, 0.1, 0.1)
                    # 手爪闭合
                    gripper_target.put(0)  # 异步，不等待
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    # 完成吸盘操作，回到放置点
                    VC.vacuum_clean_by_layer_U625_2(U625_layer_num, 0.2, 0.2, 0.1, 0.1)
                    # 放手
                    gripper_target.put(100)  # 异步，不等待
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    # time.sleep(2)
                    # 回到等待点
                    VC.vacuum_clean_by_layer_U625_3(U625_layer_num, 0.5, 0.5, 0.1, 0.1)
                    current_point = 'wait_conf'  # 当前在等待点
                else:   # 不吸盘
                    # 是否有计算结果
                    if D435_result is None:
                        print("D435线程没有计算结果")
                        U625_End = True
                        rbt_r.moveL(wait_pose, fast_v, fast_a)  # 回到等待点
                        current_point = 'wait_conf'
                        gripper_ready_event.wait()  # 等待手爪动作完成
                        gripper_ready_event.clear()  # 复位
                        return
                    # 获取U625的抓取和放置路径
                    U625_grasp_and_place_path = D435_result
                    # 接近路径
                    if current_point == 'wait_conf':    # 当前在等待点
                        U625_approach_path = U625_grasp_and_place_path['app']
                    else:   # 为了能使梅卡曼德能第一时间拍照，现在都在等待点
                        print("current_point异常，程序结束")
                        End = True
                        prepare_Mech_camera.set()
                        prepare_D435_camera.set()
                        rbt_r.red_lamp()    # 红灯亮
                        gripper_ready_event.wait()  # 等待手爪动作完成
                        gripper_ready_event.clear()  # 复位
                        return
                    rbt_r.move_jnts(U625_approach_path[0], slow_v, slow_a)  # 保险起见
                    rbt_r.move_jntspace_path(path=U625_approach_path, interval_time=1.0, control_frequency=.002,
                                             vel=fast_v, acc=fast_a,
                                             speed_gain=300,
                                             blend=0.0, toppra_vels=[fast_v]*6, toppra_accs=[fast_a]*6)
                    # 抓取低点位姿,改成从仿真中获得
                    U625_low_grasp_pose = U625_grasp_and_place_path['low_grasp_pose']
                    # print(f"U625最终抓取低点位姿:{U625_low_grasp_pose}")
                    # 抓取点位姿,改成从仿真中获得
                    U625_grasp_pose = U625_grasp_and_place_path['grasp_pose']
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    # 下移到抓取点
                    rbt_r.moveL(U625_grasp_pose, slow_v, slow_a)
                    # time.sleep(20)
                    # 张开手爪，抓取
                    gripper_target.put((80,30))  # 异步，不等待
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    gripper_target.put('width')
                    gripper_ready_event.wait()  # 等待完成
                    gripper_ready_event.clear()  # 复位
                    # 检查手爪宽度
                    width = gripper_current_width
                    print(f"U625手爪宽度:{width}")
                    U625_grasp_success = True if 25 < width < 70 else False  # 抓住时在 38 左右
                    if not U625_grasp_success:  # 没抓住，重新抓一遍
                        print("没抓住U625，重新抓")
                        # 上移3mm...、
                        rbt_r.moveL(U625_grasp_pose +np.array([0,0,0.003,0,0,0]), slow_v, slow_a)
                        # 闭合
                        gripper_target.put(0)  # 异步，不等待
                        rbt_r.moveL(U625_grasp_pose , slow_v, slow_a)   # 再回去
                        gripper_ready_event.wait()  # 等待手爪动作完成
                        gripper_ready_event.clear()  # 复位
                        FC_time = ForceControl.move(0.3,1, force = -10)    # 力控下移
                        if FC_time<=1: # 碰到底部
                            ForceControl.move(0.3,0,10)  # 力控上移一点
                        gripper_target.put(80)  # 异步，不等待
                        gripper_ready_event.wait()  # 等待手爪动作完成
                        gripper_ready_event.clear()  # 复位
                        rbt_r.moveL(U625_grasp_pose, slow_v, slow_a)
                    else:   # 抓住了
                        # 上来一点点3mm
                        rbt_r.moveL(U625_grasp_pose+np.array([0,0,0.003,0,0,0]), slow_v, slow_a)
                    # 判断是否抓住
                    gripper_target.put(80)  # 异步，不等待
                    # 到抓取低点下面1mm处(但不低于抓取点),竖直向上,moveL
                    rbt_r.moveL(U625_low_grasp_pose - np.array([0, 0, 0.001, 0, 0, 0]), slow_v, slow_a)
                    gripper_ready_event.wait()  # 等待手爪动作完成
                    gripper_ready_event.clear()  # 复位
                    gripper_target.put('width')
                    gripper_ready_event.wait()  # 等待完成
                    gripper_ready_event.clear()  # 复位
                    width = gripper_current_width
                    print(f"U625手爪宽度:{width}")

                    U625_grasp_success = True if 25 < width < 70 else False  # 抓住时在 38 左右
                    if not U625_grasp_success:  # 没抓住
                        print("没抓住U625，回到等待点，需要重新抓U625，结束当前循环")
                        back_path1 = U625_approach_path[::-1]
                        rbt_r.move_jnts(back_path1[0], slow_v, slow_a)  # 必须有
                        rbt_r.move_jntspace_path(path=back_path1, interval_time=1.0, control_frequency=.002,
                                                 vel=fast_v, acc=fast_a,
                                                 speed_gain=300,
                                                 blend=0.0, toppra_vels=[fast_v] * 6, toppra_accs=[fast_a] * 6)
                        current_point = 'wait_conf'
                    else:   # 抓住了
                        print("抓住了U625，执行放置")
                        # 放置路径,从抓取低点到放置高点
                        place_path = U625_grasp_and_place_path['place']
                        rbt_r.move_jnts(place_path[0], slow_v, slow_a)  # 必须有
                        # 检测放置区域是否有物体
                        camera_detect_time2 = time.time()
                        while True:  # 一直检测到无物体
                            # 相机拍摄RGB图
                            rgb_image = D435_2.capture_rgb(delay=0.1, show=False, save=False)
                            if not U625_detect_model.detect_exist_U625(
                                    rgb_image, x_range=(470,600), y_range=(50,180), save=False, show=False):
                                print("检测到放置区域无U1，可以放置")
                                break
                            if time.time() - camera_detect_time2 > 30:
                                End = False
                                print("等待物体放置超时，程序结束")
                                return
                            time.sleep(1)
                            print("U625放置区域有物体")
                        print(f"U625检测放置区域用时{time.time() - camera_detect_time2}")

                        rbt_r.move_jntspace_path(path=place_path, interval_time=1.0, control_frequency=.002,
                                                 vel=fast_v, acc=fast_a,
                                                 speed_gain=300,
                                                 blend=0.0, toppra_vels=[fast_v]*6, toppra_accs=[fast_a]*6)
                        # 下移到放置点，慢速
                        rbt_r.moveL(CONFIG_U625['place']['place_pose'], slow_v, slow_a)
                        # 手爪闭合释放
                        # gripper_r.close_g()
                        gripper_target.put(0)  # 异步，不等待
                        gripper_ready_event.wait()  # 等待手爪动作完成
                        gripper_ready_event.clear()  # 复位
                        U625_place_num += 1
                        print(f"已放置{U625_place_num}个U625")
                        # 返回放置高点，使用moveL
                        rbt_r.moveL(CONFIG_U625['place']['high_place_pose'],slow_v, slow_a)
                        current_point = 'U625_high_place_conf'
                        # 到下一循环再返回等待点

        print(f"[机器人线程] 第 {loop_count_0} 次循环结束")
    except Exception as d:
        print(f"[机器人线程] 发生错误: {d}")
        traceback.print_exc()
        End =True
        prepare_Mech_camera.set()  # 相机线程避免一直等待
        prepare_D435_camera.set()  # 相机线程避免一直等待

"""梅卡曼德相机线程函数"""
def Mech_thread(loop_count_0):
    try:
        print(f"[相机线程] 第 {loop_count_0} 次循环开始")
        global End, U1_pick_height, cameraclient, Mech_mask_center  # 要修改的全局变量
        global U1_need_vacuum_clean, U1_layer_num, U1_End

        prepare_Mech_camera.wait()  # 等待机器人到达等待位置,下达拍摄命令，已经知道上次抓没抓住grasp_success
        # 重置事件状态
        prepare_Mech_camera.clear()
        print("[相机线程] 梅卡曼德开始拍摄")
        grasp_success_0 = U1_grasp_success # 获取最新的U1抓取信息
        # 采集新数据
        start_time = time.time()
        color_image_0, depth_image_0 = cameraclient.capture()  # 多进程拍摄（大约1.3s）
        if color_image_0 is None:
            clear_queue(Mech_queue)  # 清空grasp_queue队列
            Mech_queue.put(None)  # 放入None表示没有结果
            End = True  # 设置结束标志为True
            return
        pcb_image_0 = mech.generate_pointcloud(color_image_0,depth_image_0,show=False,pcb_out_path=Mech_pcb_out_path) # 生成点云
        # 处理数据（使用实时采集的图像）
        if pcb_image_0 is None:    # 点云生成失败，跳出线程，可以再检测一遍
            clear_queue(Mech_queue)  # 清空grasp_queue队列
            Mech_queue.put(None)  # 放入None表示没有结果
            End = True  # 设置结束标志为True
            return
        has_mask_0, cls_id_0, pos_0, new_mask_center, angle_0 = process_image_and_pointcloud(
            color_image_0, pcb_image_0, grasp_success_0, Mech_mask_center
        )
        # 如果没有检测到掩码，发送吸盘信号
        if not has_mask_0:
            print(f"[相机线程] 第 {loop_count_0} 次循环没有检测到掩码，执行吸盘操作")
            U1_cardboard_height = mech.mean_depth_in_xy_range(pcb_image_0,[-0.1,0.1],[-0.2,0.2],CONFIG_U1['T_cam_to_world'])
            print(f"U1纸板高度:{U1_cardboard_height}")
            if U1_cardboard_height >= 0.805:    # 0.819左右
                U1_layer_num = 1
                U1_End = False
                print(f"在第1层 (高度: {U1_cardboard_height:.3f}m)")
            elif U1_cardboard_height >= 0.775:  # 0.79左右,0.794
                U1_layer_num = 2
                U1_End = False
                print(f"在第2层 (高度: {U1_cardboard_height:.3f}m)")
            else:
                U1_layer_num = 3    # 0.761左右,0.764
                print(f"在第3层 (高度: {U1_cardboard_height:.3f}m)")
                print("U1抓取完毕，抓取程序结束")
                End = True
                clear_queue(Mech_queue)  # 清空result_queue队列
                Mech_queue.put(None)
                return
            U1_need_vacuum_clean = True
            clear_queue(Mech_queue)  # 清空result_queue队列
            Mech_queue.put(None)
            return
        # 检测到掩码,更新全局变量
        Mech_mask_center = new_mask_center
        # 有检测到物体
        U1_need_vacuum_clean = False  # 吸盘标志复位。如果吸盘抓取失败则一直抓取。
        # 计算抓取路径
        # 最终物体位置
        if not grasp_success_0:   # 前一个物体抓取失败
            print("U1之前没有抓住")
            U1_pick_height -= U1_change_pick_height  # 抓取高度降低
        else:
            print("U1之前抓住了")
            U1_pick_height = CONFIG_U1['grasp']['pick_height']  # 抓取高度恢复
        # 修正物体位置
        U1_pos_0 = pos_0 + np.array([0, 0, U1_pick_height])
        print(f"U1最终位置{U1_pos_0}")
        # 抓取角度和路径,每次都从等待点出发
        if angle_0 is None:
            angle_0 = 0
        # 抓取和放置路径
        grasp_and_place_path_0 = Path_plan.U1_grasp_and_place_path(cls_id_0, U1_pos_0, angle_0, wait_rot)
        clear_queue(Mech_queue) # 清除队列结果，必须要有
        if grasp_and_place_path_0 is None:
            Mech_queue.put(None)
            End = True
            return
        # 将结果放入队列
        grasp_tuple_0 = (cls_id_0, grasp_and_place_path_0)   # 种类和抓取路径
        Mech_queue.put(grasp_tuple_0) # 放入新数据
        print(f"[相机线程] 第 {loop_count_0} 次循环结束")
        print(f"梅卡曼德相机线程计算总时间：{time.time() - start_time}")
    except Exception as e:
        print(f"[相机线程] 发生错误: {e}")
        traceback.print_exc()
        End = True

# D435相机线程程序
def D435_thread(loop_count_1):
    global U625_pick_height, U625_need_vacuum_clean, End, D435_mask_center,U625_layer_num, U625_End
    # 等待发出拍摄指令
    prepare_D435_camera.wait()
    print(f"第{loop_count_1}次D435循环开始拍摄")
    # 重置事件状态
    prepare_D435_camera.clear()
    grasp_success_1 = U625_grasp_success  # 获取最新的U625抓取信息
    D435_start_time = time.time()
    # 拍摄图片,生成点云
    color_image, depth_image, D435_pcb = D435_1.capture_from_camera(delay = 0.1, save = True, show = False)
    try:
        # 图片中箱子外面涂黑
        # 创建全黑背景图（与原图尺寸相同）
        U625_box_image = np.zeros_like(color_image)  # 全黑
        # 将箱子框内区域复制到黑图上，中心坐标（宽325，高245）
        U625_box_image[105:385, 125:525] =color_image[105:385, 125:525]
        # yolo检测, 生成掩码
        D435_last_point = D435_mask_center  # 上次的掩码中心
        mask , D435_mask_center = D435_seg_model.detect_and_save_masks(
            image_path=None,
            color_image=U625_box_image,
            last_point= D435_last_point,    # 之前掩码的位置
            grasp_success= grasp_success_1, # 之前是否抓取成功
            save=True    # 是否保存结果
        )
        if mask is None :
            print(f"循环{loop_count_1}: 相机未检测到物体，设置need_vacuum_clean标志")
            # 获取纸板在世界坐标系实际高度
            U625_cardboard_height = D435_1.mean_depth_in_xy_range(D435_pcb, [-0.2, 0.2], [-0.1, 0.1],CONFIG_U625['T_cam_to_world'])
            print(f"U625纸板高度{U625_cardboard_height}")
            if U625_cardboard_height >= 0.92:  # 0.92左右，0.934
                U625_layer_num = 1
                print(f"在第1层 (高度: {U625_cardboard_height:.3f}m)")
            elif U625_cardboard_height >= 0.89:    # 0.886左右，0.916,0.90
                U625_layer_num = 2
                print(f"在第2层 (高度: {U625_cardboard_height:.3f}m)")
            else:
                U625_layer_num = 3  # 0.86左右，0.884,0.88
                print(f"在第3层 (高度: {U625_cardboard_height:.3f}m)")
                End = True  # U625抓取完毕
                clear_queue(D435_queue)  # 清空result_queue队列
                D435_queue.put(None)
                print("U625抓取完毕，抓取程序结束")
                return

            U625_need_vacuum_clean = True # 使用吸盘吸U625箱子纸板标志
            clear_queue(D435_queue)  # 清空result_queue队列
            D435_queue.put(None)
            return
    except Exception as e:
        print(f"YOLO检测出错: {e}")
        End = True  # 准备结束程序
        clear_queue(D435_queue)  # 清空result_queue队列
        D435_queue.put(None)  # 放入None表示没有结果
        return
    # 有检测到物体
    U625_need_vacuum_clean = False  # 吸盘标志复位。如果吸取失败则一直吸取。
    # 点云裁剪
    U625_cropped_pcd = mask_cropper.main(
        ply_data = D435_pcb,
        mask_data = mask,
        color_image = color_image,
        output_path = D435_ply_path,    # 裁剪后的点云路径
        show = False,
    )
    # icp匹配
    U625_transformation, scale_factor = icp.run_rotation_only_icp(
        U625_cropped_pcd, None, STL625_PATH,
        manual_scale=1,
        use_cluster_center=True,
        use_bbox_center=True,
        eps=0.02,
        min_samples=30,
        show = False,   # 是否展示
      save = False  # 是否保存结果信息
      )
    # 物体位置
    U625_pos0 = U625_transformation[:3, 3]
    U625_pos1 = D435_1.cam_to_world(U625_pos0, CONFIG_U625['T_cam_to_world']) + np.array([0.008, -0.015, 0])  # 位置补偿
    # 最终位置
    if not grasp_success_1:  # 前一个物体抓取失败
        print("U625之前没有抓住")
        U625_pick_height -= U625_change_pick_height  # 抓取高度降低
    else:
        print("U625之前抓住了")
        U625_pick_height = CONFIG_U625['grasp']['pick_height']  # 抓取高度恢复
    U625_pos = U625_pos1 + np.array([0, 0, U625_pick_height])
    print(f"U625位置：{U625_pos}")
    # 抓取和放置路径
    U625_grasp_and_place_path = Path_plan.U625_grasp_and_place_path(wait_conf, U625_pos)
    # 放入结果
    clear_queue(D435_queue) # 清除队列结果，必须要有
    D435_queue.put(U625_grasp_and_place_path)
    print(f"D435线程总时间：{time.time()-D435_start_time}秒")

def gripper_thread(gripper_0):
    """独立手爪控制线程"""
    global gripper_current_width
    while True:
        target = gripper_target.get()   # 一直等待输入，直到有输入，其中包括输入None
        if target is None:  # 退出信号
            break
        # 执行张开/闭合动作（可能分步）
        if isinstance(target, int):
            # 只有宽度
            gripper_0.jaw_to(target)
        elif isinstance(target, tuple) and len(target) == 2:
            # (宽度, 力)
            width, force = target
            gripper_0.set_force(force)
            gripper_0.jaw_to(width)
        elif target == 'width':
            # 更新当前宽度
            with gripper_current_width_lock:  # type:ignore 线程锁，
                gripper_current_width = gripper_0.get_current_width()
        else:
            print(f"未知的手爪命令: {target}")
            continue
        # 通知主函数
        gripper_ready_event.set()   # 通知主函数手爪运动完成


if __name__ == '__main__':
    # 初始化
    # ==================== 模式设置 =======================
    move_rbt = True   # 是否连接真实机器人
    camera_detect = False   # 是否只拍照
    rbt_sim = False  # 是否开启仿真
    test = False  # 测试程序
    # ==================== 物体总层数 =======================
    U1_total_layer_num = 3
    U625_total_layer_num = 3
    # ==================== 系统配置路径 ====================
    base_dir = os.path.dirname(os.path.abspath(__file__))    # 获取当前脚本所在目录
    Mech_save_dir = os.path.join(base_dir, "images", "Mech") # 梅卡曼德图片保存路径
    D435_save_dir = os.path.join(base_dir, "images", "D435") # D435相机图片保存路径
    Mech_pcb_out_path = os.path.join(base_dir, "images", "Mech", "pointcloud.ply")  # 梅卡曼德点云输出路径(固定路径和名称)
    Mech_ply_path = os.path.join(base_dir, "images", "Mech", "output_cropped2.ply") # 梅卡曼德裁剪后点云路径(固定路径和名称)
    D435_ply_path = os.path.join(base_dir, "images", "D435", "output_cropped2.ply") # D435裁剪后点云路径(固定路径和名称)
    STL_PATH = os.path.join(base_dir, "object", "U1_2.STL") # U1倒放物体路径
    STL1_PATH = os.path.join(base_dir, "object", "u1.STL")  # U1正放物体路径
    STL625_PATH = os.path.join(base_dir, "object", "U625.STL")  # U625物体路径
    Mech_seg_path = os.path.join(base_dir, "yolo_pths", "best4.pt") # U1YOLO分割模型路径
    Mech_angle_path = os.path.join(base_dir, "yolo_pths", "best1.pt")   # U1YOLO角度模型路径
    D435_seg_path = os.path.join(base_dir, "yolo_pths", "bestutest.pt") # U625YOLO分割模型路径
    U1_in_place_path = os.path.join(base_dir, "yolo_pths", "best5.pt")   # YOLO检测是否有U1物体模型路径
    U625_in_place_path = os.path.join(base_dir, "yolo_pths", "best6.pt") # YOLO检测是否有U625物体模型路径

    # ==================== 固定点位参数 ====================
    U1_pick_height = CONFIG_U1['grasp']['pick_height']  # 抓取高度补偿,在机器人线程中也要改
    U1_change_pick_height = CONFIG_U1['grasp']['change_pick_height']  # 抓取高度变化量
    U1_place_height = CONFIG_U1['place']['place_height'] # 放置高点距离放置点的高度

    U625_pick_height = CONFIG_U625['grasp']['pick_height']  # 抓取高度补偿,在机器人线程中也要改
    U625_change_pick_height = CONFIG_U625['grasp']['change_pick_height']  # 抓取高度变化量
    # 中间等待位置，机器人刚好离开相机拍摄区域，也是丢弃位置
    wait_pos = CONFIG_U1['grasp']['wait_pos']
    wait_rot = CONFIG_U1['grasp']['wait_rot']
    wait_conf = CONFIG_U1['grasp']['wait_conf']
    wait_pose = CONFIG_U1['grasp']['wait_pose']
    # ==================== 共享数据容器 ====================
    Mech_mask_center = CONFIG_U1['grasp']['mask_center']  # 梅卡曼德掩码中心的初始化
    D435_mask_center = CONFIG_U625['grasp']['mask_center'] # D435掩码中心的初始化
    U1_place_num = 0   # 已放置的U1物体数量
    U625_place_num = 0  # 已放置的U625物体数量
    U1_grasp_success =True # U1上次是否抓取成功
    U625_grasp_success = True  # U625上次是否抓取成功
    U625_need_vacuum_clean = False # U625吸盘
    U1_need_vacuum_clean = False  # U1吸盘
    U1_layer_num = 1 # U1层数
    U625_layer_num = 1  # U625层数

    U625_End = False # U625抓取是否结束
    U1_End = False  # U1抓取是否结束
    gripper_current_width = 0  # 手爪当前实际宽度（由手爪线程更新）
    # ==================== 线程同步事件 ====================
    prepare_Mech_camera = threading.Event()  # 梅卡曼德相机"准备拍照"事件
    prepare_D435_camera = threading.Event()  # D435相机"准备拍照"事件
    gripper_ready_event = threading.Event() # 手爪是否运动完成
    # 队列
    Mech_queue = queue.Queue(maxsize=1)  # 下一个抓取路径，最多存一次数据
    D435_queue = queue.Queue(maxsize=1)  # 下一个抓取路径，最多存一次数据
    gripper_target = queue.Queue()  # 手爪期望宽度
    # ==================== 立即初始化所有模块 ====================
    # 1. 仿真环境
    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    rbt_s = cbt.UR7E(pos=np.array([0.7, 0.2, 0.7]), rotmat=rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi),
                     enable_cc=True)  # 仿真机器人
    # rbt_s.gen_meshmodel().attach_to(base)
    # 2. 障碍物模型
    obstacle_list = rbt_s.get_obstacle_list(base,False)

    # 3. 机器人模块
    print("初始化机器人模块...")
    rbt_r = ur7con.UR5Ag95X_RTDE(
        robot_ip='192.168.125.30',  # 机器人IP
        gp_port='COM5'  # 夹爪端口
    ) if move_rbt else None
    if move_rbt:
        rbt_r.green_lamp()  # 绿灯亮
    gripper_r = gripper.Ag145driver() if move_rbt else None   # 手爪控制
    # 4.力控模块
    ForceControl = fc.ForceControl(rbt_r) if move_rbt else None # 力控模块
    # 5. Yolo模型
    Mech_seg_model = YOLO.YOLODetector_objs(yolo_path=Mech_seg_path, save_dir= Mech_save_dir)  # 梅卡曼德分割
    far_angle_model = YOLO.YOLODetector_angle(yolo_path=Mech_angle_path, save_dir= Mech_save_dir)  # 梅卡曼德角度
    D435_seg_model = YOLO.YOLOToMask(model_path=D435_seg_path, save_dir= D435_save_dir,conf_threshold=0.7)  # D435分割
    U1_detect_model = YOLO.YOLO_detect_exist_U1(yolo_path=U1_in_place_path, save_dir= D435_save_dir)    # 检测有无U1
    U625_detect_model = YOLO.YOLO_detect_exist_U625(yolo_path=U625_in_place_path, save_dir=D435_save_dir)   # 检测有无U625
    # 6. 点云裁剪模块
    mask_cropper = mcpc.PointCloudCropper()
    # 7. ICP配准模块
    icp = scale.RotationOnlyICP()
    # 8. 相机模块
    mech = Mech.CaptureImage(save_directory=Mech_save_dir)
    D435_1 = D435.D435Detector(save_directory=D435_save_dir,camera_serial = '241122073898') # 箱子上方相机
    D435_2 = D435.D435Detector(save_directory=D435_save_dir,camera_serial='317222074435')   # 放置点上方相机
    # 9. 路径规划模块
    Path_plan = pp.Path_plan(rbt_s = rbt_s,obstacle_list = obstacle_list,base = base)
    # 10.双进程获取图片类
    cameraclient = sgi.CameraClient()
    # 11.吸盘模块
    VC = vc.Vacuum_Clean(rbt_r, ForceControl)
    if move_rbt:
        rbt_r.io(0, 0)  # 设置io信号,不吸气

    if test:
        camera_detect, rbt_sim, move_rbt = False, False, False
        rbt_r.move_jnts(wait_conf, 2, 2)  # 移动到等待位置
        # TCP的坐标系方向与世界坐标系方向（x轴方向相反，y轴方向相反，z轴方向相同）
        start_pose = rbt_r.rtde_r.getActualTCPPose()  # 返回 [x,y,z,rx,ry,rz]
        print(f"当前位姿:{start_pose}")
        # 目标位姿：y方向偏移 -0.05 米
        target_pose = start_pose.copy()
        target_pose[1] += 0.05
        while 1:
            rbt_r.moveL(target_pose, vel=0.1, acc=0.1)
            print(f"目标位姿:{rbt_r.rtde_r.getActualTCPPose()}")
            rbt_r.moveL(start_pose, vel=0.1, acc=0.1)
            print(f"初始位姿:{rbt_r.rtde_r.getActualTCPPose()}")

    if camera_detect:   # 是否只拍照
        # start = time.time()
        # prepare_path, mask_center, pos, high_grasp_pos = run_1_camera()   # 数据采集与计算
        # total_time=time.time() - start
        # print(f"total_time:{total_time}")
        move_rbt = False
        rbt_sim = False
        # 把点云放到仿真环境里
        D435_color_image, D435_depth_image, D435_pcb1 = D435_1.capture_from_camera(delay=0.1, save=True, show=False)
        pcd_np = np.asarray(D435_pcb1.points)  # 将点云转化为np数据
        # 点云坐标变换
        T_cam_to_world_U625 = CONFIG_U625['T_cam_to_world']  # 相机外参
        pcd = mech.transform_point_cloud(pcd_np, T_cam_to_world_U625)
        gm.gen_pointcloud(pcd,rgbas = [[0, 1, 0, .3]]).attach_to(base)  # 把变换后的点云放到仿真环境里
        # 把点云放到仿真环境里
        Mech_color_image1, Mech_depth_image1, Mech_pcb_image1 = mech.capture_and_generate_pointcloud(
            save=True, show=False, pcb_out_path=Mech_pcb_out_path
        )
        pcd_np1 = np.asarray(Mech_pcb_image1.points)  # 将点云转化为np数据
        # 点云坐标变换
        T_cam_to_world_U1 = CONFIG_U1['T_cam_to_world']  # 相机外参
        pcd1 = mech.transform_point_cloud(pcd_np1, T_cam_to_world_U1)
        gm.gen_pointcloud(pcd1, rgbas=[[0, 1, 0, .3]]).attach_to(base)  # 把变换后的点云放到仿真环境里
        base.run()

    if rbt_sim: # 是否开启仿真
        # 展示碰撞体
        rbt_s.show_cdprimit()

        if not move_rbt:    # 只仿真，机器人不动
            jnts = np.array([1.6853, - 1.1569, 1.7564, - 2.1702, - 1.5708, 1.9535])
            rbt_s.fk("arm", jnts)
            collision_info = rbt_s.is_collided(obstacle_list)
            print(collision_info)
            rbt_s.gen_meshmodel().attach_to(base)
            base.run()

    if move_rbt:    # 运行真实机器人
        print("\n运行多线程机器人模式...")
        # 创建手爪线程
        gripper_current_width = 0  # 当前实际宽度（由手爪线程更新）
        gripper_current_width_lock = threading.Lock()
        gripper_thread = threading.Thread(
            target=gripper_thread,  # 运行的函数名
            args=(gripper_r,),  # 给函数的变量
            daemon = True,   # 主程序退出时线程自动终止
        )  # 机器人线程
        gripper_thread.start()  # 手爪线程开启
        gripper_target.put(80)  # 异步，不等待
        print("机器人张开")
        if rbt_r.check_rtder_is_connected():
            print("未掉线")
        else:
            print("rtde_r重连失败,程序终止")
            sys.exit(1) # 程序终止
        current_conf = rbt_r.get_jnt_values()   # 当前关节点
        if (current_conf != wait_conf).any():   # 移动到等待位置
            start_path = Path_plan.plan_path(current_conf, wait_conf,0.1)
            if rbt_sim: # 仿真检查
                robot_mesh = robot_simulation(start_path, [0, 1, 0, 1])
                # base.run()
            if start_path is not None and len(start_path) > 5:
                rbt_r.move_jnts(start_path[0], 0.1, 0.1)    # 到初始点，保险起见
                rbt_r.move_jntspace_path(
                    path=start_path,
                    interval_time=1,
                    control_frequency=.002,
                    vel=0.1, acc=0.1,
                    speed_gain=300,
                    blend=0.0,
                    toppra_vels=[0.1] * 6,
                    toppra_accs=[0.1] * 6,
                )
        rbt_r.move_jnts(wait_conf, 0.1, 0.1)    # 保险起见
        # 运行第一次拍摄
        start_time_1 = time.time()
        success = run_1_camera()
        print(f"数据处理总时间:{time.time() - start_time_1}")
        End = False
        if not success:
            U1_End = True
            print("第一次拍摄U1失败")
        """运行连续的多线程循环，直到没有检测到掩码"""
        # 循环计数器初始化
        loop_count = 1  # 从1开始
        current_point = 'wait_conf' # 记录当前点
        gripper_ready_event.wait()  # 等待手爪运行完成
        gripper_ready_event.clear()  # 手爪事件复位
        # 连续循环，直到没有检测到掩码
        while not End:
            if rbt_sim:
                print("检查仿真")
                robot_mesh_list = []  # 存储机器人的网格模型,用于仿真
                grasp_result = Mech_queue.get(timeout=30.0)
                # 如果队列里是None，表示没有检测到掩码
                if grasp_result is not None:
                    # 解包结果
                    (cls_id, grasp_path) = grasp_result
                    if grasp_path['app'] is not None:
                        robot_mesh1 = robot_simulation(grasp_path['app'], [0, 1, 0, 0.5])
                        robot_mesh_list.append(robot_mesh1)
                        robot_mesh2 = robot_simulation(grasp_path['place'], [1, 1, 0, 0.5])
                        robot_mesh_list.append(robot_mesh2)
                    base.run()
                    clear_queue(Mech_queue)  # 清空grasp_queue队列
                    Mech_queue.put(grasp_result)  # 放入新数据
            print(f"开始第 {loop_count} 次循环")
            # 创建线程
            robot_thread = threading.Thread(
                target=thread_robot,    # 运行的函数名
                args=(loop_count,), # 给函数的变量
                name=f"RobotThread-{loop_count}"
            )  # 机器人线程
            # 添加手爪线程
            mech_thread = threading.Thread(
                target=Mech_thread,  # 运行的函数名
                args=(loop_count,),  # 给函数的变量
                name=f"CameraThread-{loop_count}"
            )   # 梅卡曼德相机线程
            d435_thread = threading.Thread(
                target=D435_thread,  # 运行的函数名
                args=(loop_count,),  # 给函数的变量
                name=f"CameraThread-{loop_count}"
            )  # D435相机线程
            # 线程开启
            robot_thread.start()  # 机器人线程启动
            d435_thread.start()  # D435相机线程启动
            time.sleep(0.1)
            mech_thread.start()  # 梅卡曼德相机线程启动
            # 等待线程结束
            robot_thread.join() # 机器人线程
            d435_thread.join()  # D435相机线程
            mech_thread.join()  # 梅卡曼德相机线程
            # 检查是否连续两次没有掩码，如果没有则退出循环
            if End:
                print(f"第{loop_count}次循环出现问题，结束循环")
                break
            # 安全检查：最多执行50次循环（防止无限循环）
            if loop_count > 100:
                print("已达到最大循环次数（100次），强制结束")
                break
            # 循环计数器加1
            loop_count += 1
            if U1_End and U625_End: # 两个物体都抓完毕
                End = True
        print(f"\n✓ 连续循环结束，共执行了 {loop_count - 1} 次循环")
        print(f"\n✓ 共放置 {U1_place_num} 个U1，{U625_place_num}个U625")

    gripper_target.put(None)  # 手爪线程停止
    gripper_thread.join()   # 等待当前手爪线程结束
    print("程序结束")
    # 结束灯
    rbt_r.red_lamp()
    rbt_s.show_cdprimit()  # 机器人展示碰撞体
    # base.run()

