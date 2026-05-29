import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
from PIL import Image
from typing import Literal, Tuple, Optional, Dict, Any, List
import open3d as o3d
from config import CONFIG_U1,CONFIG_U625


class D435Detector:
    def __init__(self, camera_type: Literal['d405', 'd435'] = 'd435', aruco_dict=None, aruco_params=None,
                 save_directory='yanpu_ur8/images/D435', camera_serial: str = None):
        self.camera_type = camera_type
        self.save_directory = save_directory
        self.camera_serial = camera_serial

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
        self.depth_scale = None

        self.initialize_camera()    # 初始化


    @staticmethod
    def get_connected_cameras() -> List[Dict[str, str]]:
        """
        获取所有连接的 RealSense 相机信息

        Returns:
            相机信息列表，每个元素包含 serial 和 name
        """
        ctx = rs.context()
        devices = ctx.query_devices()

        camera_list = []
        for dev in devices:
            info = {
                'serial': dev.get_info(rs.camera_info.serial_number),
                'name': dev.get_info(rs.camera_info.name),
                'firmware': dev.get_info(rs.camera_info.firmware_version)
            }
            camera_list.append(info)
            print(f"检测到相机: {info['name']} (序列号: {info['serial']}, 固件: {info['firmware']})")

        return camera_list

    def initialize_camera(self):
        """初始化相机（只在第一次调用）"""
        if self.pipeline is None:
            self.pipeline = rs.pipeline()
            config = rs.config()
            # 如果指定了序列号，则绑定到特定相机
            if self.camera_serial:
                config.enable_device(self.camera_serial)
                print(f"正在连接到相机 (序列号: {self.camera_serial})...")
            else:
                print("未指定序列号，将连接第一个可用相机...")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            try:
                self.pipeline_profile = self.pipeline.start(config)
                self.device = self.pipeline_profile.get_device()
                # 更新实际使用的序列号
                if not self.camera_serial:
                    self.camera_serial = self.device.get_info(rs.camera_info.serial_number)
                    print(f"已连接到相机 (序列号: {self.camera_serial})")

                self.align = rs.align(rs.stream.color)
                # 预热相机，丢弃前几帧
                for _ in range(5):
                    self.pipeline.wait_for_frames()
                self.depth_sensor = self.device.first_depth_sensor()
                self.depth_scale = self.depth_sensor.get_depth_scale()
                time.sleep(3)
                print(f"深度缩放因子: {self.depth_scale} m/unit")
            except Exception as e:
                print(f"相机初始化失败: {e}")
                raise

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

            if keep_invalid:    # 移除无效点
                # 创建有效点掩码，深度范围在0.90到1.1之间
                valid_mask = (z > 0.9) & (z < depth_trunc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

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
                # 不移除无效点（原始逻辑）深度范围在0.9到1.1
                valid_mask = (z > 0.9) & (z < depth_trunc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
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
            # z轴翻转
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

            if visualize:   # 可视化
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

    def capture_from_camera(self,delay: float = 3.0,
                            save = True,
                            show = False ):
        """
        从RealSense相机捕获图像并生成点云，返回PLY文件路径

        Args:
            delay: 自动捕获延迟时间（秒）
            save: 保存
            show: 展示
        Returns:
            color_image:彩色图像
            depth_image:深度图像
            ply_image:点云图像

        """
        self.initialize_camera()    # 如果掉线，则重新初始化
        try:
            print(f"将在 {delay} 秒后自动捕获图像...")
            start_time = time.time()
            captured_color_image = None
            captured_depth_image = None
            pointcloud = None
            while time.time() - start_time < delay:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue
                if show:
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
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if color_frame and depth_frame:
                captured_color_image = np.asanyarray(color_frame.get_data())
                captured_depth_image = np.asanyarray(depth_frame.get_data())
                # 生成并保存点云
                print("正在生成点云...")
                pointcloud = self.generate_complete_pointcloud(
                    captured_color_image,
                    captured_depth_image,
                    depth_scale=self.depth_scale,
                    depth_trunc = 1.1,  # 最大深度,超出部分点云去除
                    visualize=show,
                    keep_invalid=True  # 保留无效点
                )
                if save:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    color_path = os.path.join(self.save_directory, f'rgb_{timestamp}.jpg')
                    depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')
                    cv2.imwrite(color_path, captured_color_image)
                    Image.fromarray(captured_depth_image.astype(np.uint16)).save(depth_path)
                    print(f"已保存图像: {color_path}, {depth_path}在{self.save_directory}中")
                    # 保存点云为PLY文件
                    if pointcloud is not None:
                        ply_filename = f'pointcloud_{timestamp}.ply'
                        ply_file_path = os.path.join(self.save_directory, ply_filename)
                        o3d.io.write_point_cloud(ply_file_path, pointcloud, write_ascii=False)  # type:ignore
                        print(f"已保存PLY文件: {ply_file_path}在{self.save_directory}中")
        finally:
            cv2.destroyAllWindows()
        return captured_color_image, captured_depth_image, pointcloud

    def capture_from_camera_first(self, delay: float = 3.0,show = False,save = True):
        '''
            只拍RGB图和深度图
        '''
        self.initialize_camera()    # 如果掉线，重新初始化
        captured_color_image = None
        captured_depth_image = None
        print(f"将在 {delay} 秒后自动捕获图像...")
        start_time = time.time()
        while time.time() - start_time < delay:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            if show:
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
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame and depth_frame:
            captured_color_image = np.asanyarray(color_frame.get_data())
            captured_depth_image = np.asanyarray(depth_frame.get_data())
            if save:
                # 保存图像
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
                depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')
                cv2.imwrite(color_path, captured_color_image)
                Image.fromarray(captured_depth_image.astype(np.uint16)).save(depth_path)
                print(f"已保存图像: {color_path}, {depth_path}")
        # self.pipeline.stop()
        cv2.destroyAllWindows()

        return captured_color_image, captured_depth_image


    def capture_from_camera_second(self , captured_color_image , captured_depth_image, show = False, save = True):

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # pipeline_profile = self.pipeline.start(config)
        # device = pipeline_profile.get_device()
        # depth_sensor = device.first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # pointcloud = None
        xx = time.time()
        depth_scale = 0.0010000000474974513
        pointcloud = self.generate_complete_pointcloud(
            captured_color_image,
            captured_depth_image,
            depth_scale=depth_scale,
            visualize=show,
            keep_invalid=True  # 保留无效点
        )
        print(time.time()-xx)
        print('//')
        if save:
            # 保存点云为PLY文件
            ply_filename = f'pointcloud_{timestamp}.ply'
            ply_file_path = os.path.join(self.save_directory, ply_filename)
            # 保存PLY文件
            o3d.io.write_point_cloud(ply_file_path, pointcloud, write_ascii=False)
            print(f"已保存PLY文件: {ply_file_path}")

        cv2.destroyAllWindows()

        return  ply_file_path, pointcloud


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

    """只拍RGB图"""

    def capture_rgb(self, delay: float = 0.5, show=True, save=False) -> Optional[np.ndarray]:

        self.initialize_camera()  # 如果掉线，则重新初始化

        print(f"将在 {delay} 秒后自动捕获图像...")
        start_time = time.time()
        while time.time() - start_time < delay:
            frames = self.pipeline.wait_for_frames()
            # 获取彩色帧
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
        print("正式捕获图像...")
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            captured_color_image = np.asanyarray(color_frame.get_data())
        else:
            print("警告：未能获取rgb图像")
            return None
        if save:
            # 保存图像
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            color_path = os.path.join(self.save_directory, f'color_image2_{timestamp}.jpg')
            print(f"RGB图像保存于{color_path}")
            # 保存彩色图像（BGR格式）
            cv2.imwrite(color_path, captured_color_image)
        if show:  # 展示RGB图像
            # 显示RGB图像
            cv2.imshow('Captured Image', captured_color_image)
            cv2.waitKey(1000)  # 显示1秒
            cv2.destroyAllWindows()

        return captured_color_image

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
            return depths.mean()
        else:
            print("警告：所有点都在原点！")
            return None

    def cam_to_world(self, point_cam, t_cam_to_world):
        '''
        将相机坐标系的点的坐标转换为世界坐标系的点的坐标
        输入:
            point_cam:相机坐标系下的位置
            t_cam_to_world:外参矩阵
        输出:
            point_world:世界坐标系下的位置
        '''
        point_cam = np.asarray(point_cam).reshape(1, 3)  # 确保是二维
        # print(f"point_cam: {point_cam}")
        # 转换为齐次坐标 (N, 4)
        points_hom = np.hstack([point_cam, np.ones((1, 1))])
        points_world_hom = (t_cam_to_world @ points_hom.T).T  # 得到 (N, 4),.T表示转置
        point_world = points_world_hom[0, :3]
        return point_world

    def mean_depth_in_xy_range(self, pcd: o3d.geometry.PointCloud, x_range, y_range, t_cam_to_world):
        """
        计算点云在指定 x, y 矩形范围内的有效点的平均深度 (z值)
        参数:
            pcd: Open3D 点云对象
            x_range: (x_min, x_max)相机坐标系下点云区域
            y_range: (y_min, y_max)相机坐标系下点云区域
            t_cam_to_world: 坐标变换矩阵
        返回:
            mean_depth: 世界坐标系下平均深度 (米)，若无有效点则返回 None
        """
        points = np.asarray(pcd.points)
        # 排除原点 (0,0,0)
        valid_mask = ~np.all(points == 0, axis=1)
        # 加入 x, y 范围约束
        x_min, x_max = x_range
        y_min, y_max = y_range
        in_box_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                      (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        final_mask = valid_mask & in_box_mask
        selected_points = points[final_mask]
        if len(selected_points) == 0:
            print("警告：指定范围内没有有效点")
            return None
        depths = selected_points[:, 2]
        mean_depth = depths.mean()
        print(f"在 x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}] 范围内共 {len(selected_points)} 个有效点")
        print(f"平均深度: {mean_depth:.3f} m")
        point_cam = [0, 0, mean_depth]
        point_world = self.cam_to_world(point_cam, t_cam_to_world)

        return point_world[-1]  # 世界坐标系的深度

    def transform_point_cloud(self,pcd_np, transform_matrix):
        """
        将点云从相机坐标系变换到世界坐标系
        :param pcd_np: (N, 3) numpy 数组，相机坐标系下的点坐标
        :param transform_matrix: (4, 4) 齐次变换矩阵，相机->世界
        :return: (N, 3) 世界坐标系下的点坐标
        """
        # 转为齐次坐标
        N = pcd_np.shape[0]
        pcd_hom = np.hstack((pcd_np, np.ones((N, 1))))  # (N, 4)
        pcd_world_hom = (transform_matrix @ pcd_hom.T).T  # (N, 4)
        return pcd_world_hom[:, :3]


if __name__ == "__main__":
    # 使用原始字符串避免转义问题
    save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"

    # # 创建检测器
    # detector = D435Detector(camera_type='d435', save_directory=save_dir,)
    # # 获取序列号
    # camera_list = detector.get_connected_cameras()
    # print(camera_list)
    # [{'serial': '241122073898', 'name': 'Intel RealSense D435', 'firmware': '5.13.0.55'}, {'serial': '317222074435', 'name': 'Intel RealSense D435', 'firmware': '5.13.0.55'}]
    # 箱子上方D435相机
    D435_1 = D435Detector(camera_type='d435', save_directory=save_dir, camera_serial = '241122073898')
    # D435_1.capture_rgb(0.5,True,True)
    # 放置点上方D435相机
    D435_2 = D435Detector(camera_type='d435', save_directory=save_dir, camera_serial='317222074435')
    # D435_2.capture_rgb(0.5, True, True)

    # image = detector.capture_rgb(0.5,True,True)
    # # 示例：自动捕获
    # print("=== 自动捕获模式 ===")
    # color_img, depth_img, ply_path = detector.capture_from_camera(
    #     delay=3,
    #     save = True,
    #     show = False
    # )
    # ply = detector.capture_from_camera_second(color_img,depth_img)
    # if color_img is not None:
    #     print(f"彩色图像尺寸: {color_img.shape}")
    #     cv2.imshow('Captured Color Image', color_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # if depth_img is not None:
    #     print(f"深度图像尺寸: {depth_img.shape}")
    #

    # ply_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\pointcloud_20260527-140547.ply"
    # if ply_path is not None:
    #     print(f"PLY文件路径: {ply_path}")
    #     print(f"文件存在: {os.path.exists(ply_path)}")
    #
    #     # 加载点云并分析
    #     if os.path.exists(ply_path):
    #         pcd = o3d.io.read_point_cloud(ply_path)
    #         D435_1.analyze_pointcloud_distribution(pcd)
    #
    #         # 可视化点云
    #         o3d.visualization.draw_geometries([pcd],
    #                                           window_name=f"Point Cloud with Invalid Points",
    #                                           width=800,
    #                                           height=600,
    #                                           point_show_normal=False)

    # 测试纸板在第几层
    # 拍摄图片
    # color_image, depth_image ,D435_pcb= detector.capture_from_camera(delay=0.1,save = True,show = False)
    # # 获取纸板在世界坐标系实际高度
    # U625_cardboard_height = detector.mean_depth_in_xy_range(D435_pcb, [-0.2, 0.2], [-0.1, 0.1],
    #                                                     CONFIG_U625['T_cam_to_world'])
    # print(f"U625_cardboard_height:{U625_cardboard_height}")
    # # 比较纸板高度判断在第几层,可能会错
    # if U625_cardboard_height < 0.88:  # 0.86左右
    #     print("在第3层")
    # elif U625_cardboard_height < 0.9:    # 0.886左右，0.912，0.916，
    #     print("在第2层")
    # else:   # 0.92左右，0.934
    #     print("在第1层")
# 第一层0.92，0.9208046148657696，0.9207378915440554，0.9207378915440554
# 第二层0.8863522796643943，0.886117356886964，0.8857220277030848
# 第三层0.8634936603261563，0.8626835575298855，0.8643907404035733

    # color_image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\color_image_20260514-200505.jpg"
    # depth_image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\depth_image_20260516-181752.png"
    # color_image = cv2.imread(color_image_path)
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)   # 单通道，保持uint16
    # pcd = detector.generate_complete_pointcloud(color_image= color_image, depth_image= depth_image,
    #                                  depth_scale = 0.001, visualize = True,
    #                                  depth_trunc = 1.1, keep_invalid = True)
