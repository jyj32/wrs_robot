import os
import cv2
from check_object import ObjectDetector
from realsense_camera import RealSenseCamera
import numpy as np
# 拍图片数据集

if __name__ == '__main__':
    # 物体类型
    object = 'nut'
    # 图片保存目录
    if object == 'blot':    # 螺栓
        src_dataset_dir = r'dataset\bolts\src'  # 原图路径
        dst_dataset_dir = r'dataset\bolts\dst'  # 仿射变换后路径
        thresh_dataset_dir = r'dataset\bolts\thresh'  # 二值化后路径
    elif object == 'gasket':    # 垫片
        src_dataset_dir = r'dataset\gasket\src'
        dst_dataset_dir = r'dataset\gasket\dst'  # 仿射变换后路径
        thresh_dataset_dir = r'dataset\gasket\thresh'  # 二值化后路径
    else:   # 螺母
        src_dataset_dir = r'dataset\nut\src'
        dst_dataset_dir = r'dataset\nut\dst'  # 仿射变换后路径
        thresh_dataset_dir = r'dataset\nut\thresh'  # 二值化后路径
    # 检查目录是否存在
    os.makedirs(src_dataset_dir, exist_ok=True)
    os.makedirs(dst_dataset_dir, exist_ok=True)
    os.makedirs(thresh_dataset_dir, exist_ok=True)
    # 启动相机
    camera = RealSenseCamera(camera_type='d435', save_directory=src_dataset_dir)  # 启动RealSense D435相机
    detector = ObjectDetector(obj_type=object)  # type:ignore
    camera.start()
    camera.interactive_capture()    # 相机拍照
    # 初始化收集坐标的字典（键为标记ID，值为坐标列表）
    all_coords = {0: [], 1: [], 2: [], 3: []}
    total_valid_frames = 0  # 成功检测到完整4个标记的图片数量
    # 处理图片
    for root, dirs, files in os.walk(src_dataset_dir):
        for file in files:
            if file.endswith('.png'):   # 使用无损压缩png
                try:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dataset_dir, file)
                    thresh_path = os.path.join(thresh_dataset_dir, file)
                    if not os.path.exists(dst_path):
                        src_image = cv2.imread(src_path)
                        copy_src = src_image.copy()
                        # 检测ArUco标记,获取位置
                        aruco_px, img_marked = detector.detect_aruco_pixels(src_image, draw=False)
                        # 只有当4个标记全检测到时，才收集坐标
                        if aruco_px and all(k in aruco_px for k in [0, 1, 2, 3]):
                            for marker_id in [0, 1, 2, 3]:
                                # aruco_px[marker_id] 是一个 (x, y) 元组
                                all_coords[marker_id].append(aruco_px[marker_id])
                            total_valid_frames += 1
                        else:
                            print(f"图片 {file} 缺少构建坐标系所需的 ArUco 标记 (0,1,2,3)，跳过坐标收集")
                            continue  # 缺少标记时不进行透视变换，直接处理下一张
                        warped_image, M = detector.warp_to_marker_frame(copy_src, aruco_px, output_size=(460, 340))
                        cv2.imwrite(dst_path, warped_image)
                        print(f"已保存仿射变换处理后的图片:{dst_path}")
                    else:
                        warped_image = cv2.imread(dst_path)
                except Exception as e:
                    print(f"处理图片{os.path.join(root, file)}时发生错误:{e}")
    # --- 循环结束后，计算每个标记的平均坐标 ---
    if total_valid_frames == 0:
        print("没有找到任何包含完整4个ArUco标记的有效图片，无法计算平均坐标")
    else:
        avg_coords = {}
        for marker_id, coord_list in all_coords.items():
            if coord_list:
                # 分别计算 x 和 y 的平均值
                avg_x = np.mean([c[0] for c in coord_list])
                avg_y = np.mean([c[1] for c in coord_list])
                avg_coords[marker_id] = (avg_x, avg_y)
            else:
                avg_coords[marker_id] = None
                print(f"警告：标记 {marker_id} 没有任何有效坐标数据")

        print(f"\n=== 基于 {total_valid_frames} 张图片计算的平均坐标 ===")
        for marker_id, coord in avg_coords.items():
            if coord:
                print(f"ID {marker_id}: ({coord[0]}, {coord[1]})")
            else:
                print(f"ID {marker_id}: 无数据")

