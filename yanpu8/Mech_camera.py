import os
import open3d as o3d
import cv2
import time
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect
import tifffile
import numpy as np
from typing import Optional

from yanpu_ur8.config import CONFIG_U1


# 梅卡曼德相机拍照
class CaptureImage(object):
    # 初始化
    def __init__(self, save_directory = 'yanpu_ur8/images/Mech'):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()
        self.save_directory = save_directory
        find_and_connect(self.camera, 0) # 连接相机
    # 拍RGB图像
    def capture_rgb(self, save=True, show=False):
        frame_2d = Frame2D()
        show_error(self.camera.capture_2d(frame_2d))
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()   # 灰度图
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()    # BGR 格式的彩色图像
        else:
            # 不支持的颜色类型，可能报错
            print("不支持的图像颜色类型")
            return None
        if image2d is None:
            print("获取图像失败")
            return None
        if show:
            cv2.imshow("RGB Image", image2d.data())
            cv2.waitKey(5000)  # 等待5秒自动关闭、或按键直接关闭
            cv2.destroyAllWindows()  # 关闭所有窗口
        if save: # 保存图像
            os.makedirs(self.save_directory, exist_ok=True) # 确保目录存在
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"color_image_{timestamp}.jpg"
            file_path = os.path.join(self.save_directory, file_name)
            cv2.imwrite(file_path, image2d.data())
            print(f"保存RGB图像: {file_name}于{self.save_directory}")
        return image2d.data()

    # 深度图
    def capture_depth_map(self, save=True, show=False):
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d)) # 打印错误代码及其描述信息
        depth_map = frame3d.get_depth_map()
        if show:
            cv2.imshow("Depth Map", depth_map.data())
            cv2.waitKey(5000)  # 等待5秒自动关闭、或按键直接关闭
            cv2.destroyAllWindows()  # 关闭所有窗口
        if save: # 保存图像
            depth_data = depth_map.data()   # 32 位浮点深度图，推荐保存为 TIFF
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"depth_{timestamp}.tiff"
            file_path = os.path.join(self.save_directory, file_name)
            tifffile.imwrite(file_path, depth_data)  # 保存为 32 位浮点 TIFF
            print(f"保存深度图（32位TIFF）: {file_path}")

        return depth_map.data()
    # 点云图
    def capture_point_cloud(self, save=True, show=False):
        frame_all_2d_3d = Frame2DAnd3D()
        # 捕获一帧 2D+3D 数据
        show_error(self.camera.capture_2d_and_3d(frame_all_2d_3d))
        if save:
            os.makedirs(os.path.join(self.save_directory), exist_ok=True)   # 确保目录存在
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"PointCloud_{timestamp}.ply"    # 文件名
            file_path = os.path.join(self.save_directory, file_name)
            # 保存无纹理点云
            show_error(frame_all_2d_3d.frame_3d().save_untextured_point_cloud(FileFormat_PLY, file_path))
            print(f"保存点云图片: {file_name}于{self.save_directory}")
            if show:    # 展示点云图
                if os.path.exists(file_path):
                    pcd = o3d.io.read_point_cloud(file_path)    # type: ignore[arg-type]
                    if pcd.has_points():
                        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud",
                                                          width=800,
                                                          height=600,
                                                          )
                    else:
                        print("点云文件无有效点，无法显示")
                else:
                    print("点云文件未找到，无法显示")

        return frame_all_2d_3d

    # 根据RGB图片和深度图生成点云图
    def generate_pointcloud(self, color_image: np.ndarray, depth_image: np.ndarray,
                                 depth_scale: float = 0.001, show: bool = False,
                                 depth_trunc: float = 3.0, keep_invalid: bool = True,
                                pcb_out_path:str = None,
                                 ) -> Optional[o3d.geometry.PointCloud]:
        """
        生成与RGB图像像素点数量一致的完整点云，保留所有像素点，无效点设为(0,0,0)

        Args:
            color_image: 彩色图像 (BGR格式)
            depth_image: 深度图像
            depth_scale: 深度缩放因子 (米/深度单位)
            show: 是否可视化点云
            depth_trunc: 深度截断距离，单位米
            keep_invalid: 是否保留无效点：深度过大或为负数

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
            # 梅卡曼德相机内参，从官方代码里的util里的get_camera_intrinsics.py输出
            camera_matrix = np.array([[2422.091560244942, 0, 947.874304916491],
                                    [0, 2422.206493636134, 587.0741386399137],
                                    [0, 0, 1]])
            # 计算3D坐标
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            # 反投影公式：x = (u - cx) * z / fx, y = (v - cy) * z / fy
            z = depth_flat.copy()  # 创建深度副本

            # 计算所有点的3D坐标
            x = (u_coords_flat - cx) * z / fx
            y = (v_coords_flat - cy) * z / fy

            if keep_invalid:
                # 创建有效点掩码
                valid_mask = (z > 0) & (z < depth_trunc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

                # 统计无效点
                # invalid_count = np.sum(~valid_mask)
                # total_count = len(z)
                # print(f"总像素数量: {total_count}")
                # print(f"无效点数量: {invalid_count}")
                # print(f"无效点比例: {invalid_count / total_count * 100:.2f}%")

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
            # print(f"生成点云数量: {len(points)}")
            # 创建open3d点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_flat)
            # 绕x轴旋转180度，从下往上看
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

            if show:
                o3d.visualization.draw_geometries([pcd],    # type: ignore
                                                  width=800,
                                                  height=600,
                                                  point_show_normal=False)
            ply_file_path = pcb_out_path    # 点云输出路径
            # 保存PLY文件
            o3d.io.write_point_cloud(ply_file_path, pcd, write_ascii=False) # type: ignore
            print(f"已保存PLY文件: {ply_file_path}")
            return pcd
        except Exception as e:
            print(f"生成点云时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_rgb_and_depth(self, save=True, show=False):
        """
        同时捕获配准的RGB图像和深度图像。
        返回 (rgb_data, depth_data)，均为 numpy 数组。
        rgb_data: BGR 格式，uint8
        depth_data: float32 深度值，单位通常为米（需根据实际确认）
        """
        # 同时获取RGB和深度信息
        frame_all = Frame2DAnd3D()
        show_error(self.camera.capture_2d_and_3d(frame_all))

        # 获取彩色图像
        frame_2d = frame_all.frame_2d()
        if frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            rgb_obj = frame_2d.get_color_image()    # 灰度图
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            rgb_obj = frame_2d.get_gray_scale_image()   # 彩色图
        else:
            print("不支持的图像颜色类型")
            return None, None
        if rgb_obj is None:
            print("获取RGB图像失败")
            return None, None

        # 获取深度图像
        frame_3d = frame_all.frame_3d()
        depth_map = frame_3d.get_depth_map()
        if depth_map is None:
            print("获取深度图像失败")
            return None, None

        rgb_data = rgb_obj.data()      # BGR 格式的 numpy 数组
        depth_data = depth_map.data()  # float32 深度数组
        if show:
            # 只显示 RGB
            cv2.namedWindow("RGB_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RGB_image", 960, 600)  # 设置窗口大小
            cv2.imshow("RGB_image", rgb_data)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        if save:
            os.makedirs(self.save_directory, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 保存 RGB
            rgb_filename = f"color_image_{timestamp}.jpg"
            rgb_path = os.path.join(self.save_directory, rgb_filename)
            cv2.imwrite(rgb_path, rgb_data)
            print(f"保存RGB图像: {rgb_path}")
            # # 保存深度（32位TIFF）
            # depth_filename = f"depth_{timestamp}.tiff"
            # depth_path = os.path.join(self.save_directory, depth_filename)
            # tifffile.imwrite(depth_path, depth_data)
            # print(f"保存深度图像: {depth_path}")
        return rgb_data, depth_data

    def capture_and_generate_pointcloud(self, save = True, show=False,
                                         pcb_out_path=r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech\pointcloud.ply",
                                         depth_scale=0.001, depth_trunc=3.0, keep_invalid=True):
        """
        同时捕获配准的RGB和深度图像，并生成点云。

        Args:
            save: 是否保存RGB和深度图像文件（默认True）。
            show: 是否显示RGB和深度图像预览（默认False）。
            depth_scale: 深度缩放因子（米/深度单位）。如果为None，自动根据深度值范围判断（若最大值>10则视为毫米，设为0.001；否则视为米，设为1.0）。
            depth_trunc: 深度截断距离，单位米。
            keep_invalid: 是否保留无效点（坐标设为(0,0,0)）。

        Returns:
            (rgb_data, depth_data, ply_file_path) 的元组；如果失败则返回(None, None, None)。
        """
        # 同时捕获RGB和深度图像
        rgb, depth = self.capture_rgb_and_depth(save=save, show=show)
        if rgb is None or depth is None:
            print("捕获图像失败，无法生成点云")
            return None, None, None

        # 生成点云（并自动保存PLY文件）
        ply_data = self.generate_pointcloud(
            color_image=rgb,
            depth_image=depth,
            depth_scale=depth_scale,
            show=show,
            depth_trunc=depth_trunc,
            keep_invalid=keep_invalid,
            pcb_out_path=pcb_out_path
        )
        return rgb, depth, ply_data

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

    def mean_depth_in_xy_range(self, pcd, x_range, y_range, t_cam_to_world):
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

if __name__ == '__main__':

    Mech_camera = CaptureImage(save_directory = r'images/Mech')
    # if find_and_connect(Mech_camera.camera, 0):
        # RGB_image = Mech_camera.capture_rgb(True,True)
        # Depth_image = Mech_camera.capture_depth_map(True,True)
        # ply_file = Mech_camera.capture_point_cloud(True, True)
        # if ply_file is not None:
        # RGB_image, Depth_image = Mech_camera.capture_rgb_and_depth(save=True, show=False)
        # ply_file_path = Mech_camera.generate_pointcloud(RGB_image, Depth_image, show=True)
        # RGB_image, Depth_image, ply_file = Mech_camera.capture_and_generate_pointcloud(save=True,show=True)
    # Mech_camera.camera.disconnect()    # 相机关闭

    # # 检查纸板高度
    RGB_image, Depth_image, ply_file = Mech_camera.capture_and_generate_pointcloud(save=True,pcb_out_path=r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech\pointcloud.ply",
                                                                                   show=True)
    # U1_cardboard_height = Mech_camera.mean_depth_in_xy_range(ply_file, [-0.1,0.1], [-0.2,0.2], CONFIG_U1['T_cam_to_world'])
    # print(f"U1纸板高度:{U1_cardboard_height}")
    # # 比较纸板高度判断在第几层,很可能会错
    # if U1_cardboard_height < 0.775:    # 0.761左右,0.764
    #     print("在第3层")
    # elif U1_cardboard_height < 0.805:  # 0.79左右,0.794
    #     print("在第2层")
    # else:   # 0.819左右,0.825
    #     print("在第1层")

# 第1层:0.8197817374550582,0.8197932344289134,0.8190690742067668
# 第2层:0.7898642862660914，0.7898614709504399,0.7899270439102384
# 第3层:0.7612091165018531，0.7612132139379615，0.7612382838261373
#     pcd_path = r'E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech\pointcloud.ply'
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     pcd = o3d.visualization.draw_geometries([pcd],    # type: ignore
#                                                       width=800,
#                                                       height=600,
#                                                       point_show_normal=False)