import time
from check_object import ObjectDetector
from grasp_executor import GraspPlanner, GraspExecutorRTDE
from realsense_camera import RealSenseCamera
from config import CONFIG
import robot_con.ur.ur7_dh50_rtde as ur7e_con
import robot_sim.end_effectors.gripper.dh50.dh50 as dh50
import robot_sim.robots.ur7e.ur7e as ur7e
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
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
        return_angle : bool, 是否返回角度信息

    Returns:
        rot_mat_z : 3x3 旋转矩阵
        [可选] angle_deg : 角度（单位度，范围 0~360）
    """
    vec = np.array(vec[:2], dtype=np.float64)   # 只取前两个数，组成平面向量
    if np.linalg.norm(vec) < 1e-8:
        raise ValueError("输入 vec 的模长为0，无法定义方向")

    # vec /= np.linalg.norm(vec)  # 单位化
    # y_axis = np.array([0.0, 1.0])  # 正Y轴方向
    #
    # # 点积和角度
    # dot = np.clip(np.dot(y_axis, vec), -1.0, 1.0)   # 单位向量与y轴向量点积
    # angle_rad = np.arccos(dot)  # 弧度制夹角,范围[0，pi]
    # angle_deg = np.degrees(angle_rad)   # 角度制夹角
    #
    # # 顺时针为正角度
    # angle_deg = angle_deg if vec[0] < 0 else -angle_deg
    # print(f"绕z轴旋转{angle_deg}角度")
    # theta = np.radians(angle_deg)
    # # 构造绕 Z 轴的旋转矩阵（右手系）
    # rot_mat_z = np.array([
    #     [np.cos(theta), -np.sin(theta), 0],
    #     [np.sin(theta), np.cos(theta), 0],
    #     [0, 0, 1]
    # ])
    # print(f"绕 Z 轴的旋转矩阵:{rot_mat_z}")
    # if return_angle:
    #     return rot_mat_z, angle_deg
    # else:
    #     return rot_mat_z

    angle_rad = -np.arctan2(vec[0], vec[1])  # 单位向量与y轴向量的角度，范围[-pi，pi]
    print(f"绕z轴旋转{angle_rad}弧度")
    # 构造绕 Z 轴的旋转矩阵（右手系）
    rot_mat_z = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    print(f"绕 Z 轴的旋转矩阵:{rot_mat_z}")
    if return_angle:
        angle_deg = np.degrees(angle_rad)
        return rot_mat_z, angle_deg
    else:
        return rot_mat_z

def check_and_planning_thread(image, object_detector, grasp_planner, obj_type, result_queue):
    place_pos = CONFIG[obj_type]['place_pos']
    place_rot = CONFIG[obj_type]['place_rot']
    # 目标检测
    single_obj, standing_obj, stack_obj = object_detector.run_on_image(image, draw=False)

    if len(single_obj) != 0:
        if obj_type == 'blot':
            # 抓取旋转矩阵
            grasp_rot_0 = rm.rotmat_from_axangle([1, 0, 0], np.pi/10).dot(CONFIG['robot']['default_rot'])    # 倾斜30度
            grasp_rot = get_z_rotation_matrix_from_vec(single_obj[0][1][:2], return_angle=False).dot(grasp_rot_0)
        else:
            # grasp_rot = rm.rotmat_from_axangle([1, 0, 0], np.pi) @ rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
            grasp_rot = CONFIG['robot']['default_rot']
        # 抓取物体路径规划
        step2_result = grasp_planner.grasp_and_place_single_obj(
            rbt_start_pos=CONFIG['robot']['camera_pos'],
            grasp_pos=single_obj[0][0],
            grasp_rot=grasp_rot,
            place_pos=place_pos,
            place_rot=place_rot)
        result_kind = 'single'
    elif len(standing_obj) != 0:    # 直立状态
        step2_result = grasp_planner.split_stand_objs(
            rbt_start_pos=CONFIG['robot']['camera_pos'],
            obj_pos=standing_obj[0][0],
            grip_vec=standing_obj[0][1])
        result_kind = 'single_standing'
    elif len(stack_obj) != 0:   # 堆叠状态
        step2_result = grasp_planner.split_stack_objs(
            rbt_start_pos=CONFIG['robot']['camera_pos'],
            obj_pos=stack_obj[0][0],
            grip_vec=stack_obj[0][1])
        result_kind = 'stack'
    else:   # 没有检测到物体
        step2_result, result_kind = None, None
    result_queue.put((step2_result, result_kind))

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    move_rbt = True # 是否运行真实机器人
    obj_type = 'blot'   # 设置物体种类
    obj_size = 'M2' # 设置物体尺寸
    rbt_r = ur7e_con.UR5Ag95X_RTDE(robot_ip='192.168.125.30',
                                   gp_port='COM5') if move_rbt else None
    gripper_s = dh50.Dh50()
    rbt_s = ur7e.UR7E()
    rbt_s.gen_meshmodel().attach_to(base)
    gasket_obj = cm.CollisionModel(r'.\stl_model\gasket_c.STL')
    blot_obj = cm.CollisionModel(r'.\stl_model\blot_c.STL')
    nut_obj = cm.CollisionModel(r'.\stl_model\nut_c.STL')
    obj_dict = {'gasket': gasket_obj, 'blot': blot_obj, 'nut': nut_obj}

    plate_obj = cm.CollisionModel(r'.\stl_model\plate_c.STL')
    plate_obj.set_pos(np.array([0.075, 0.55, -0.02]) + np.array([0.085, 0.115, 0]))
    plate_obj.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    plate_obj.set_rgba([0, 0, 1, 0.5])
    plate_obj.attach_to(base)
    gm.gen_frame(pos=np.array([0.7125, -0.0875, 0]) + np.array([0.17, 0, 0]),
                 rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)).attach_to(base)
    grasp_planner = GraspPlanner(base, robot_sim=rbt_s, gripper_sim=gripper_s)  # 路径规划器

    grasp_executor = GraspExecutorRTDE(robot_real=rbt_r, move_rbt=move_rbt)
    # 相机初始化
    camera = RealSenseCamera(camera_type='d435', save_directory='Data_Intel_Realsense_d435')
    # yolo模型初始化
    object_detector = ObjectDetector(obj_type=obj_type, size = obj_size)    # type: ignore

    result_queue = queue.Queue()    # 队列
    # 从盒子中抓取物体
    step1_confs = grasp_planner.get_objs_from_box(
        rbt_start_pos=CONFIG['robot']['camera_pos'],
        box_pos=CONFIG['box']['box_pos'], place_pos=CONFIG['box']['place_pos'], approach_dis=CONFIG['box']['app_dist'])
    grasp_executor.get_objs_from_box_executor(step1_confs)  # 从盒中抓取物体，回到拍摄点
    # base.run()
    time.sleep(10)
    camera.start()  # 相机启动
    image = camera.capture()  # 相机拍摄
    # 相机线程
    planning_t = threading.Thread(target=check_and_planning_thread,
                                  args=(image, object_detector, grasp_planner, obj_type, result_queue))
    planning_t.start()

    # 循环抓取
    while True:
        step2_results, result_type = result_queue.get()  # 等待计算抓取路径
        if step2_results is None:
            print("没有检测到物体")
            break
        if result_type == 'single': # 单个物体可抓
            print("检测到单个物体")
            # 执行抓取
            grasp_executor.grasp_single_obj_executor(step2_results) # 执行单个抓取
            time.sleep(0.5)
            image = camera.capture()    # 相机拍摄
            # 路径规划
            planning_t = threading.Thread(target=check_and_planning_thread,
                                          args=(image, object_detector, grasp_planner, obj_type, result_queue))
            planning_t.start()
            # 执行放置
            grasp_executor.place_single_obj_executor(step2_results)
        else:
            if result_type == 'single_standing':
                print("检测到直立物体")
                grasp_executor.split_stand_objs_executor(step2_results)    # 执行推倒
                time.sleep(0.5)
            elif result_type == 'stack':   # 堆叠状态
                print("检测到堆叠物体")
                grasp_executor.split_stack_objs_executor(step2_results)    # 执行分开
                time.sleep(0.5)
            image = camera.capture()    # 相机拍摄
            # 路径规划
            planning_t = threading.Thread(target=check_and_planning_thread,
                                          args=(image, object_detector, grasp_planner, obj_type, result_queue))
            planning_t.start()
        # base.run()

    # 回到home点
    if move_rbt:
        rbt_r.move_jnts(np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi]), 0.5)

    base.run()  # 程序运行完，展示之前经过的状态
