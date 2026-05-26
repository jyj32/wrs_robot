import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
from PIL import Image
from typing import Literal, Tuple, Optional, Dict, Any
import open3d as o3d


class GasketDetector:
    def __init__(self, camera_type: Literal['d405', 'd435'] = 'd405', aruco_dict=None, aruco_params=None,
                 save_directory='Data_Intel_Realsense_D405'):
        self.camera_type = camera_type
        self.save_directory = save_directory

        # 确保保存目录存在
        os.makedirs(self.save_directory, exist_ok=True)

        if self.camera_type.lower() == 'd405':
            self.camera_matrix = np.array([[434.43981934, 0.0, 318.67144775],
                                           [0.0, 434.35751343, 241.73374939],
                                           [0.0, 0.0, 1.0]])
            self.dist_coeffs = np.array([[-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724]])
        elif self.camera_type.lower() == 'd435':
            # self.camera_matrix = np.array([[610.0, 0.0, 323.0],
            #                                [0.0, 610.0, 245.0],
            #                                [0.0, 0.0, 1.0]])
            self.camera_matrix = np.array([[605.492,  0,     326.025],
                                           [0,      604.954, 243.011],
                                           [0,      0,       1     ]])
            self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            print("注意：D435使用默认相机参数，建议进行相机标定获取准确参数")
        else:
            raise ValueError('Either d405 or d435')

        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco_params or aruco.DetectorParameters()

        # 点云处理相关
        self.pipeline = None
        self.pc = rs.pointcloud()
        self.colorizer = rs.colorizer()

    def generate_complete_pointcloud(self, color_image: np.ndarray, depth_image: np.ndarray,
                                     depth_scale: float = 0.001, visualize: bool = False,
                                     depth_trunc: float = 3.0, keep_invalid: bool = True) -> Optional[
        o3d.geometry.PointCloud]:
        """
        生成与RGB图像像素点数量一致的完整点云，保留所有像素点，无效点设为(0,0,0)

        Args:
            color_image: 彩色图像 (BGR格式)
            depth_image: 深度图像
            depth_scale: 深度缩放因子 (米/深度单位)
            visualize: 是否可视化点云
            depth_trunc: 深度截断距离，单位米
            keep_invalid: 是否保留无效点（设为(0,0,0)）

        Returns:
            open3d点云对象，包含所有像素点
        """
        if color_image is None or depth_image is None:
            print("彩色图或深度图为空")
            return None

        try:
            # 获取图像尺寸
            height, width = depth_image.shape[:2]

            # 验证尺寸一致性
            if color_image.shape[0] != height or color_image.shape[1] != width:
                print("警告：彩色图和深度图尺寸不一致，将调整彩色图尺寸")
                color_image = cv2.resize(color_image, (width, height))

            # 将BGR转换为RGB
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # 创建像素坐标网格
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

            # 展平坐标
            u_coords_flat = u_coords.flatten()
            v_coords_flat = v_coords.flatten()

            # 获取深度值并展平
            depth_flat = depth_image.flatten().astype(np.float32) * depth_scale

            # 获取颜色值并展平
            colors_flat = color_image_rgb.reshape(-1, 3).astype(np.float32) / 255.0

            # 计算3D坐标
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            # 反投影公式：x = (u - cx) * z / fx, y = (v - cy) * z / fy
            z = depth_flat.copy()  # 创建深度副本

            # 计算所有点的3D坐标
            x = (u_coords_flat - cx) * z / fx
            y = (v_coords_flat - cy) * z / fy

            if keep_invalid:
                # 创建有效点掩码
                valid_mask = (z > 0) & (z < depth_trunc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

                # 统计无效点
                invalid_count = np.sum(~valid_mask)
                total_count = len(z)
                print(f"总像素数量: {total_count}")
                print(f"无效点数量: {invalid_count}")
                print(f"无效点比例: {invalid_count / total_count * 100:.2f}%")

                # 将无效点的坐标设为(0, 0, 0)
                x[~valid_mask] = 0.0
                y[~valid_mask] = 0.0
                z[~valid_mask] = 0.0

                # 对于无效点，使用默认颜色（灰色）或原始颜色
                # colors_flat[~valid_mask] = [0.5, 0.5, 0.5]  # 可选项：将无效点颜色设为灰色
            else:
                # 移除无效点（原始逻辑）
                valid_mask = (z > 0) & (z < depth_trunc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x = x[valid_mask]
                y = y[valid_mask]
                z = z[valid_mask]
                colors_flat = colors_flat[valid_mask]

            # 创建点云数组
            points = np.column_stack((x, y, z))

            print(f"生成点云数量: {len(points)}")

            if keep_invalid:
                # 统计(0,0,0)点的数量
                zero_points = np.sum(np.all(points == 0, axis=1))
                print(f"位于原点(0,0,0)的点数量: {zero_points}")
                print(f"原点点比例: {zero_points / len(points) * 100:.2f}%")

            # 创建open3d点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_flat)


            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

            if visualize:
                o3d.visualization.draw_geometries([pcd],
                                                  window_name="3D Point Cloud (保留所有点)",
                                                  width=800,
                                                  height=600,
                                                  point_show_normal=False)

            return pcd

        except Exception as e:
            print(f"生成点云时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_from_camera(self, generate_pointcloud: bool = True,
                            method: str = 'manual',
                            auto_capture: bool = False,
                            delay: float = 3.0) -> Tuple[Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[str]]:
        """
        从RealSense相机捕获图像并生成点云，返回PLY文件路径

        Args:
            generate_pointcloud: 是否生成点云
            method: 点云生成方法 'manual' 或 'rs_pointcloud'
            auto_capture: 是否自动捕获（True=自动延迟后捕获，False=手动按键捕获）
            delay: 自动捕获延迟时间（秒）

        Returns:
            (color_image, depth_image, ply_file_path)
            返回捕获的彩色图像、深度图像和PLY文件路径
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动管道并获取设备信息
        pipeline_profile = self.pipeline.start(config)
        time.sleep(5)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # 获取深度缩放因子
        depth_scale = depth_sensor.get_depth_scale()
        print(f"深度缩放因子: {depth_scale} m/unit")

        align_to = rs.stream.color
        align = rs.align(align_to)

        captured_color_image = None
        captured_depth_image = None
        ply_file_path = None

        try:
            if auto_capture:
                print(f"将在 {delay} 秒后自动捕获图像...")
                start_time = time.time()

                while time.time() - start_time < delay:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    # 显示实时预览
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    combined = np.hstack((color_image, depth_colormap))
                    cv2.imshow('Auto Capture Preview', combined)
                    cv2.waitKey(1)

                print("正在捕获图像...")
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if color_frame and depth_frame:
                    captured_color_image = np.asanyarray(color_frame.get_data())
                    captured_depth_image = np.asanyarray(depth_frame.get_data())

                    # 保存图像
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
                    depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')
                    cv2.imwrite(color_path, captured_color_image)
                    Image.fromarray(captured_depth_image.astype(np.uint16)).save(depth_path)
                    print(f"已保存图像: {color_path}, {depth_path}")

                    # 生成并保存点云
                    if generate_pointcloud:
                        print("正在生成点云...")
                        pointcloud = None

                        if method == 'manual':
                            pointcloud = self.generate_complete_pointcloud(
                                captured_color_image,
                                captured_depth_image,
                                depth_scale=depth_scale,
                                visualize=False,
                                keep_invalid=True  # 保留无效点
                            )
                        elif method == 'rs_pointcloud':
                            pointcloud = self.generate_pointcloud_from_depth_frame(
                                color_frame,
                                depth_frame,
                                depth_scale=depth_scale,
                                visualize=False
                            )

                        if pointcloud is not None:
                            # 保存点云为PLY文件
                            ply_filename = f'pointcloud_{timestamp}.ply'
                            ply_file_path = os.path.join(self.save_directory, ply_filename)

                            # 保存PLY文件
                            o3d.io.write_point_cloud(ply_file_path, pointcloud, write_ascii=False)
                            print(f"已保存PLY文件: {ply_file_path}")

                            # 同时保存为其他格式
                            pcd_filename = f'pointcloud_{timestamp}'
                            saved_files = self.save_point_cloud(
                                pointcloud,
                                pcd_filename,
                                save_formats=['pcd', 'txt']
                            )

                            # 生成点云截图
                            # self.save_pointcloud_screenshot(pointcloud, timestamp)

                cv2.destroyAllWindows()

            else:
                # 手动捕获模式
                print("相机启动，按 Enter 键拍照并生成点云，按 'v' 键可视化点云，按 's' 键只保存图像，按 'q' 退出...")
                print(f"点云生成方法: {method}")

                while True:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # 显示深度图
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )

                    # 并排显示彩色图和深度图
                    combined = np.hstack((color_image, depth_colormap))
                    cv2.imshow('Color (Left) | Depth (Right) - Press Enter to capture', combined)

                    key = cv2.waitKey(1)

                    if key == 13:  # Enter 键
                        timestamp = time.strftime("%Y%m%d-%H%M%S")

                        # 保存图像
                        color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
                        depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')
                        cv2.imwrite(color_path, color_image)
                        Image.fromarray(depth_image.astype(np.uint16)).save(depth_path)
                        print(f"已保存图像: {color_path}, {depth_path}")

                        captured_color_image = color_image.copy()
                        captured_depth_image = depth_image.copy()

                        # 生成并保存点云
                        if generate_pointcloud:
                            print("正在生成点云...")
                            pointcloud = None

                            if method == 'manual':
                                pointcloud = self.generate_complete_pointcloud(
                                    captured_color_image,
                                    captured_depth_image,
                                    depth_scale=depth_scale,
                                    visualize=False,
                                    keep_invalid=True  # 保留无效点
                                )
                            elif method == 'rs_pointcloud':
                                pointcloud = self.generate_pointcloud_from_depth_frame(
                                    color_frame,
                                    depth_frame,
                                    depth_scale=depth_scale,
                                    visualize=False
                                )

                            if pointcloud is not None:
                                # 保存点云为PLY文件
                                ply_filename = f'pointcloud_{timestamp}.ply'
                                ply_file_path = os.path.join(self.save_directory, ply_filename)

                                # 保存PLY文件
                                o3d.io.write_point_cloud(ply_file_path, pointcloud, write_ascii=False)
                                print(f"已保存PLY文件: {ply_file_path}")

                                # 同时保存为其他格式
                                pcd_filename = f'pointcloud_{timestamp}'
                                saved_files = self.save_point_cloud(
                                    pointcloud,
                                    pcd_filename,
                                    save_formats=['pcd', 'txt']
                                )

                                # 生成点云截图
                                self.save_pointcloud_screenshot(pointcloud, timestamp)

                        break  # 捕获完成后退出循环

                    elif key == ord('v'):  # 'v' 键可视化当前帧的点云
                        print("可视化当前帧的点云...")
                        if method == 'manual':
                            temp_pcd = self.generate_complete_pointcloud(
                                color_image,
                                depth_image,
                                depth_scale=depth_scale,
                                visualize=True,
                                keep_invalid=True  # 保留无效点
                            )
                        elif method == 'rs_pointcloud':
                            temp_pcd = self.generate_pointcloud_from_depth_frame(
                                color_frame,
                                depth_frame,
                                depth_scale=depth_scale,
                                visualize=True
                            )

                    elif key & 0xFF == ord('q'):
                        break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

        return captured_color_image, captured_depth_image, ply_file_path



    def capture_from_camera_first(self, generate_pointcloud: bool = True,
                            method: str = 'manual',
                            auto_capture: bool = False,
                            delay: float = 3.0) -> Tuple[Optional[np.ndarray],
    Optional[np.ndarray]]:
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动管道并获取设备信息
        pipeline_profile = self.pipeline.start(config)
        time.sleep(10)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # 获取深度缩放因子
        depth_scale = depth_sensor.get_depth_scale()
        print(f"深度缩放因子: {depth_scale} m/unit")

        align_to = rs.stream.color
        align = rs.align(align_to)

        captured_color_image = None
        captured_depth_image = None
        ply_file_path = None

        print(f"将在 {delay} 秒后自动捕获图像...")
        start_time = time.time()

        while time.time() - start_time < delay:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 显示实时预览
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow('Auto Capture Preview', combined)
            cv2.waitKey(1)

        print("正在捕获图像...")
        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame and depth_frame:
            captured_color_image = np.asanyarray(color_frame.get_data())
            captured_depth_image = np.asanyarray(depth_frame.get_data())

            # 保存图像
            # timestamp = time.strftime("%Y%m%d-%H%M%S")
            # color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
            # depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')
            # cv2.imwrite(color_path, captured_color_image)
            # Image.fromarray(captured_depth_image.astype(np.uint16)).save(depth_path)
            # print(f"已保存图像: {color_path}, {depth_path}")


        self.pipeline.stop()
        cv2.destroyAllWindows()

        return captured_color_image, captured_depth_image


    def capture_from_camera_second(self , captured_color_image , captured_depth_image) -> Tuple[
    Optional[str]]:
        """
        从RealSense相机捕获图像并生成点云，返回PLY文件路径

        Args:
            generate_pointcloud: 是否生成点云
            method: 点云生成方法 'manual' 或 'rs_pointcloud'
            auto_capture: 是否自动捕获（True=自动延迟后捕获，False=手动按键捕获）
            delay: 自动捕获延迟时间（秒）

        Returns:
            (color_image, depth_image, ply_file_path)
            返回捕获的彩色图像、深度图像和PLY文件路径
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline_profile = self.pipeline.start(config)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        pointcloud = None


        pointcloud = self.generate_complete_pointcloud(
            captured_color_image,
            captured_depth_image,
            depth_scale=depth_scale,
            visualize=False,
            keep_invalid=True  # 保留无效点
        )

        # 保存点云为PLY文件
        ply_filename = f'pointcloud_{timestamp}.ply'
        ply_file_path = os.path.join(self.save_directory, ply_filename)
        # 保存PLY文件
        o3d.io.write_point_cloud(ply_file_path, pointcloud, write_ascii=False)
        print(f"已保存PLY文件: {ply_file_path}")

        # 同时保存为其他格式
        pcd_filename = f'pointcloud_{timestamp}'
        saved_files = self.save_point_cloud(
            pointcloud,
            pcd_filename,
            save_formats=['pcd', 'txt']
        )
        cv2.destroyAllWindows()
        self.pipeline.stop()
        cv2.destroyAllWindows()

        return  ply_file_path


    # 修改generate_pointcloud_from_depth_frame方法以保留无效点
    def generate_pointcloud_from_depth_frame(self, color_frame: rs.frame, depth_frame: rs.frame,
                                             depth_scale: float = 0.001, visualize: bool = False,
                                             keep_invalid: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """使用RealSense的pointcloud对象生成点云，保留无效点"""
        try:
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            colors = np.asanyarray(color_frame.get_data())
            colors_rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)

            # 获取图像尺寸
            height, width = colors.shape[:2]

            if keep_invalid:
                # 检查是否有点缺失
                if len(vertices) < height * width:
                    print(f"警告：点云数量({len(vertices)})少于像素数量({height * width})")
                    # 创建全尺寸的点云数组
                    full_vertices = np.zeros((height * width, 3), dtype=np.float32)
                    full_colors = np.zeros((height * width, 3), dtype=np.float32)

                    # 这里需要知道哪些像素对应哪些点
                    # 由于rs.pointcloud不提供映射信息，我们回退到手动方法
                    print("RS pointcloud方法无法保留所有像素点，切换到手动方法...")
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    return self.generate_complete_pointcloud(
                        color_image, depth_image, depth_scale, visualize, keep_invalid=True
                    )

            colors_flat = colors_rgb.reshape(-1, 3).astype(np.float32) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            if len(vertices) == len(colors_flat):
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
            else:
                default_colors = np.ones((len(vertices), 3)) * 0.5
                pcd.colors = o3d.utility.Vector3dVector(default_colors)

            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

            if visualize:
                o3d.visualization.draw_geometries([pcd],
                                                  window_name="3D Point Cloud from RS PointCloud",
                                                  width=800,
                                                  height=600,
                                                  point_show_normal=False)

            return pcd

        except Exception as e:
            print(f"使用RS pointcloud生成点云时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 其他方法保持不变...
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, filename: str,
                         save_formats: list = ['pcd', 'txt']) -> dict:
        """保存点云到PCD和TXT格式"""
        saved_files = {}

        for fmt in save_formats:
            try:
                if fmt == 'pcd':
                    filepath = os.path.join(self.save_directory, f"{filename}.pcd")
                    o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)
                    saved_files['pcd'] = filepath

                elif fmt == 'txt':
                    filepath = os.path.join(self.save_directory, f"{filename}.txt")
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors)
                    data = np.hstack((points, colors))
                    np.savetxt(filepath, data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f',
                               header='x y z r g b', comments='')
                    saved_files['txt'] = filepath

                print(f"已保存点云 ({fmt}): {filepath}")

            except Exception as e:
                print(f"保存{fmt}格式点云时出错: {e}")

        return saved_files

    # 其他方法保持不变...
    def capture_images(self, show_preview: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从RealSense相机拍摄并返回RGB和深度图片"""
        if self.pipeline is None:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)

        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("无法获取图像帧")
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 显示预览
            if show_preview:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow('RGB Image', color_image)
                cv2.imshow('Depth Colormap', depth_colormap)
                print("按任意键关闭预览窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return color_image, depth_image

        except Exception as e:
            print(f"拍摄图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def analyze_pointcloud_distribution(self, pcd: o3d.geometry.PointCloud):
        """分析点云分布情况，特别关注(0,0,0)点"""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        print("\n=== 点云分布分析 ===")
        print(f"总点数: {len(points)}")

        # 统计位于原点的点
        zero_points = np.sum(np.all(points == 0, axis=1))
        print(f"位于原点(0,0,0)的点数量: {zero_points}")
        print(f"原点点比例: {zero_points / len(points) * 100:.2f}%")

        # 统计有效点（非原点）
        valid_mask = ~np.all(points == 0, axis=1)
        valid_points = points[valid_mask]

        if len(valid_points) > 0:
            print(f"有效点数（非原点）: {len(valid_points)}")
            print(f"有效点比例: {len(valid_points) / len(points) * 100:.2f}%")

            # 深度统计
            depths = valid_points[:, 2]
            print(f"深度范围: [{depths.min():.3f}, {depths.max():.3f}] m")
            print(f"深度均值: {depths.mean():.3f} m")
            print(f"深度标准差: {depths.std():.3f} m")

            # 坐标范围
            print(f"X范围: [{valid_points[:, 0].min():.3f}, {valid_points[:, 0].max():.3f}] m")
            print(f"Y范围: [{valid_points[:, 1].min():.3f}, {valid_points[:, 1].max():.3f}] m")
        else:
            print("警告：所有点都在原点！")


if __name__ == "__main__":
    # 使用原始字符串避免转义问题
    save_dir = r"E:\py_project\wrsrobot\wrs_shu\grasp_point\image"

    # # 创建检测器
    # detector = GasketDetector(camera_type='d435', save_directory=save_dir)
    #
    # # 示例：自动捕获
    # print("=== 自动捕获模式 ===")
    # color_img, depth_img, ply_path = detector.capture_from_camera(
    #     generate_pointcloud=True,
    #     method='manual',
    #     auto_capture=True,
    #     delay=3.0
    # )
    #
    # if color_img is not None:
    #     print(f"彩色图像尺寸: {color_img.shape}")
    #     cv2.imshow('Captured Color Image', color_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # if depth_img is not None:
    #     print(f"深度图像尺寸: {depth_img.shape}")
    #
    # if ply_path is not None:
    #     print(f"PLY文件路径: {ply_path}")
    #     print(f"文件存在: {os.path.exists(ply_path)}")
    #
    #     # 加载点云并分析
    #     if os.path.exists(ply_path):
    #         pcd = o3d.io.read_point_cloud(ply_path)
    #         detector.analyze_pointcloud_distribution(pcd)
    #
    #         # 可视化点云
    #         o3d.visualization.draw_geometries([pcd],
    #                                           window_name=f"Point Cloud with Invalid Points",
    #                                           width=800,
    #                                           height=600,
    #                                           point_show_normal=False)

    detector = GasketDetector(camera_type='d435', save_directory=save_dir)

    # 直接使用相机捕获RGB图像
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        time.sleep(2)

        # 捕获一帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            rgb_path = os.path.join(save_dir, f'rgb_image_{timestamp}.jpg')
            cv2.imwrite(rgb_path, color_image)
            print(f"已保存RGB图像: {rgb_path}")

            cv2.imshow('RGB Image', color_image)
            cv2.waitKey(0)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()