import time

from check_object import ObjectDetector
from grasp_executor import GraspPlanner
from grasp_executor import GraspExecutor
from realsense_camera import RealSenseCamera

import robot_con.gofa_con.gofa_con as gofa_con
import drivers.devices.dh.maingripper as dh_r
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76
import robot_sim.robots.gofa5.gofa5 as gf5
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import numpy as np
import threading
import queue


def get_z_rotation_matrix_from_vec(vec, return_angle=False):
    """
    计算从正Y轴旋转到vec所需的绕Z轴旋转矩阵。
    顺时针为正方向。

    Parameters:
        vec : array-like, 2D or 3D vector (只使用前两个元素)
        return_angle : bool, 是否返回角度和方向信息

    Returns:
        rot_mat_z : 3x3 旋转矩阵
        [可选] angle_deg : 角度（单位度，范围 0~360）
        [可选] direction : "CW"/"CCW"
    """
    vec = np.array(vec[:2], dtype=np.float64)
    if np.linalg.norm(vec) < 1e-8:
        raise ValueError("输入 vec 的模长为0，无法定义方向")

    vec /= np.linalg.norm(vec)
    y_axis = np.array([0.0, 1.0])  # 正Y轴方向

    # 点积和角度
    dot = np.clip(np.dot(y_axis, vec), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    # 顺时针为正角度
    angle_deg = angle_deg if vec[0] < 0 else -angle_deg

    theta = np.radians(angle_deg)

    # 构造绕 Z 轴的旋转矩阵（右手系）
    rot_mat_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    if return_angle:
        return rot_mat_z, angle_deg
    else:
        return rot_mat_z


def check_and_planning_thread(image, object_detector, grasp_planner, obj_type, result_queue):
    if obj_type == 'gasket':
        place_pos = np.array([0.7375, 0.5125, 0.045])
        place_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
    elif obj_type == 'blot':
        place_pos = np.array([0.7, 0.45, 0.11])
        place_rot = (rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
                     @ rm.rotmat_from_axangle([1, 0, 0], -np.pi / 4))
    else:
        place_pos = np.array([0.7, -0.5, 0.04])
        place_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)

    single_obj, standing_obj, stack_obj = object_detector.run_on_image(image, draw=False)
    if len(single_obj) != 0:
        grasp_rot = get_z_rotation_matrix_from_vec(single_obj[0][1][:2], return_angle=False).dot(
            rm.rotmat_from_axangle([1, 0, 0], np.pi))

        step2_result = grasp_planner.grasp_and_place_single_obj(rbt_start_pos=np.array([0.713, 0.063, 0.18]),
                                                                grasp_pos=single_obj[0][0],
                                                                grasp_rot=grasp_rot,
                                                                place_pos=place_pos,
                                                                place_rot=place_rot)
        result_type = 'single'
    elif len(standing_obj) != 0:
        step2_result = grasp_planner.split_stack_or_stand_objs(rbt_start_pos=np.array([0.713, 0.063, 0.18]),
                                                               obj_pos=standing_obj[0][0],
                                                               grip_vec=standing_obj[0][1])
        result_type = 'standing'
    elif len(stack_obj) != 0:
        step2_result = grasp_planner.split_stack_or_stand_objs(rbt_start_pos=np.array([0.713, 0.063, 0.18]),
                                                               obj_pos=stack_obj[0][0], grip_vec=stack_obj[0][1])
        result_type = 'stack'
    else:
        step2_result, result_type = None, None
    result_queue.put((step2_result, result_type))


if __name__ == '__main__':
    move_rbt = True
    obj_type = 'nut'
    gripper_r = dh_r.MainGripper(port="com3", force=30, speed=30) if move_rbt else None
    rbt_r = gofa_con.GoFaArmController(toggle_debug=False) if move_rbt else None
    gripper_s = dh76.Dh76(fingertip_type='r_76')

    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    rbt_s = gf5.GOFA5()
    rbt_s.gen_meshmodel().attach_to(base)
    gasket_obj = cm.CollisionModel(r'.\stl_model\gasket_c.STL')
    blot_obj = cm.CollisionModel(r'.\stl_model\blot_c.STL')
    nut_obj = cm.CollisionModel(r'.\stl_model\nut_c.STL')
    obj_dict = {'gasket': gasket_obj, 'blot': blot_obj, 'nut': nut_obj}

    plate_obj = cm.CollisionModel(r'.\stl_model\plate_c.STL')
    plate_obj.set_pos(np.array([0.7125, -0.0875, 0]) + np.array([0.085, 0.115, 0]))
    plate_obj.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    plate_obj.set_rgba([0, 0, 1, 0.5])
    plate_obj.attach_to(base)
    gm.gen_frame(pos=np.array([0.7125, -0.0875, 0]) + np.array([0.17, 0, 0]),
                 rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)).attach_to(base)

    grasp_planner = GraspPlanner(base)
    grasp_executor = GraspExecutor(move_rbt=move_rbt, grip_r=gripper_r, rbt_r=rbt_r)
    camera = RealSenseCamera(camera_type='d405', save_directory='Data_Intel_Realsense_d435')
    object_detector = ObjectDetector(obj_type=obj_type)

    result_queue = queue.Queue()
    # step1_confs = grasp_planner.get_objs_from_box(rbt_start_pos=np.array([0.715, 0.063, 0.18]),
    #                                               box_pos=np.array([0.95, 0.45, 0]))
    # grasp_executor.get_objs_from_box_executor(step1_confs)

    camera.start()

    planning_t = threading.Thread(target=check_and_planning_thread,
                                  args=(camera.capture(), object_detector, grasp_planner, obj_type, result_queue))
    planning_t.start()

    while True:
        step2_results, result_type = result_queue.get()  # 阻塞等待路径
        if step2_results is None:
            break

        if result_type == 'single':
            grasp_executor.grasp_single_obj_executor(step2_results)
            time.sleep(0.5)
            image = camera.capture()
            planning_t = threading.Thread(target=check_and_planning_thread,
                                          args=(image, object_detector, grasp_planner, obj_type, result_queue))
            planning_t.start()
            grasp_executor.place_single_obj_executor(step2_results)
        elif result_type == 'standing' or result_type == 'stack':
            grasp_executor.split_stack_or_stand_objs_executor(step2_results)
            time.sleep(0.5)
            image = camera.capture()
            planning_t = threading.Thread(target=check_and_planning_thread,
                                          args=(image, object_detector, grasp_planner, obj_type, result_queue))
            planning_t.start()

    if move_rbt:
        rbt_r.move_j(np.array([0., 0., 0., 0., 0., 0.]), speed_n=300)

    base.run()

