import open3d as o3d
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List


class PointCloudCropper:
    """点云裁剪器，使用图像掩码裁剪3D点云"""

    def __init__(self, downsample_method: str = 'resize'):
        """
        初始化点云裁剪器

        Args:
            downsample_method: 降采样方法 ('resize' 或 'grid_sample')
        """
        self.downsample_method = downsample_method
        self.original_pcd = None
        self.cropped_pcd = None
        self.mask_info = {}
        self.image_info = {}

    def downsample_image_to_match_points(self, image: np.ndarray,
                                         target_points_count: int) -> Tuple[np.ndarray, float]:
        """
        降采样图像使其像素数与点云点数匹配

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            target_points_count: 目标像素数（点云点数）

        Returns:
            downsampled_image: 降采样后的图像
            scale_factor: 缩放因子
        """
        h, w = image.shape[:2]
        current_pixel_count = h * w

        if current_pixel_count == target_points_count:
            print(f"图像像素数与点云点数已匹配")
            return image, 1.0

        # 计算缩放因子
        scale_factor = np.sqrt(target_points_count / current_pixel_count)

        # 计算新尺寸
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # print(f"原始图像尺寸: {w}x{h} = {current_pixel_count} 像素")
        # print(f"目标像素数: {target_points_count}")
        # print(f"缩放因子: {scale_factor:.4f}")
        # print(f"新图像尺寸: {new_w}x{new_h} = {new_w * new_h} 像素")

        if self.downsample_method == 'resize':
            # 使用OpenCV resize
            if len(image.shape) == 3:
                downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        elif self.downsample_method == 'grid_sample':
            # 使用网格采样（更精确）
            x = np.linspace(0, w - 1, new_w)
            y = np.linspace(0, h - 1, new_h)
            xv, yv = np.meshgrid(x, y)

            # 双线性插值
            if len(image.shape) == 3:
                downsampled = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
                for c in range(image.shape[2]):
                    downsampled[:, :, c] = ndimage.map_coordinates(
                        image[:, :, c],
                        [yv.ravel(), xv.ravel()],
                        order=1
                    ).reshape(new_h, new_w)
            else:
                downsampled = ndimage.map_coordinates(
                    image,
                    [yv.ravel(), xv.ravel()],
                    order=1
                ).reshape(new_h, new_w)

        else:
            raise ValueError(f"未知的降采样方法: {self.downsample_method}")

        return downsampled, scale_factor

    def crop_with_downsampling(self, output_path: str = "output_cropped.ply") -> Optional[o3d.geometry.PointCloud]:
        """
        通过降采样图像使像素与点云匹配，然后裁剪点云

        Args:
            output_path: 输出文件路径

        Returns:
            裁剪后的点云
        """
        # print("=" * 60)
        print("方法: 降采样图像使像素与点云匹配")

        if self.original_pcd is None:
            raise ValueError("请先加载数据")

        # 降采样掩码
        self.mask_downsampled, mask_scale = self.downsample_image_to_match_points(
            self.original_mask, self.point_count
        )
        # 二值化降采样后的掩码
        self.binary_mask = self.mask_downsampled > 128
        mask_pixel_count = np.sum(self.binary_mask)
        # print(f"降采样后掩码中前景像素: {mask_pixel_count}")

        # 存储信息
        self.mask_info['downsampled'] = self.mask_downsampled
        self.mask_info['binary'] = self.binary_mask

        # 降采样彩色图像
        if self.original_pcd.has_colors() and self.color_image is not None:
            self.color_downsampled, color_scale = self.downsample_image_to_match_points(
                self.color_image, self.point_count
            )

            # 验证尺寸匹配
            h_mask, w_mask = self.binary_mask.shape
            h_color, w_color, _ = self.color_downsampled.shape

            if h_mask != h_color or w_mask != w_color:
                print(f"警告: 掩码({w_mask}x{h_mask})和彩色图像({w_color}x{h_color})尺寸不匹配")
                # 调整彩色图像尺寸以匹配掩码
                self.color_downsampled = cv2.resize(self.color_downsampled, (w_mask, h_mask))

            self.image_info['downsampled'] = self.color_downsampled
        else:
            self.color_downsampled = None

        # 检查点云和降采样后的图像像素一一对应
        if len(self.points) == self.binary_mask.size:
            print("点云点数与降采样图像像素数匹配 ✓")

            # 展平掩码
            mask_flat = self.binary_mask.flatten()
            self.selected_indices = np.where(mask_flat)[0]

            if len(self.selected_indices) == 0:
                print("掩码中没有前景区域")
                return None

            # 创建裁剪后的点云
            cropped_points = self.points[self.selected_indices]
            self.cropped_pcd = o3d.geometry.PointCloud()
            self.cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)

            # 应用颜色
            self._apply_colors_to_cropped_pcd()

            # 保存点云
            o3d.io.write_point_cloud(output_path, self.cropped_pcd)

            print(f"\n裁剪完成!")
            # print(f"原始点数: {self.point_count}")
            # print(f"裁剪后点数: {len(cropped_points)}")
            # print(f"保留比例: {len(cropped_points) / self.point_count:.2%}")
            return self.cropped_pcd
        else:
            print(f"错误: 点云点数({self.point_count})与降采样图像像素数({self.binary_mask.size})不匹配")
            return None

    def _apply_colors_to_cropped_pcd(self):
        """为裁剪后的点云应用颜色"""
        if self.original_pcd.has_colors() and self.color_downsampled is not None:
            colors_flat = self.color_downsampled.reshape(-1, 3) / 255.0
            cropped_colors = colors_flat[self.selected_indices]
            self.cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)
        elif self.original_pcd.has_colors():
            # 使用原始点云颜色
            colors = np.asarray(self.original_pcd.colors)
            cropped_colors = colors[self.selected_indices]
            self.cropped_pcd.colors = o3d.utility.Vector3dVector(cropped_colors)

    def crop_with_camera_projection(self, output_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        使用简化的相机投影模型裁剪点云
        """
        if self.original_pcd is None:
            raise ValueError("请先加载数据")

        print("=" * 60)
        print("方法: 相机投影匹配")
        print("=" * 60)

        # 加载掩码
        binary_mask = self.original_mask > 128
        h, w = binary_mask.shape

        # 简化的投影：假设点云的XY大致对应图像的XY
        x_min, y_min = self.points[:, :2].min(axis=0)
        x_max, y_max = self.points[:, :2].max(axis=0)

        # 将点云坐标映射到图像坐标
        points_2d = np.zeros((len(self.points), 2), dtype=int)
        points_2d[:, 0] = ((self.points[:, 0] - x_min) / (x_max - x_min) * (w - 1)).astype(int)
        points_2d[:, 1] = ((self.points[:, 1] - y_min) / (y_max - y_min) * (h - 1)).astype(int)

        # 筛选在掩码内的点
        selected_indices = []
        for i, (x, y) in enumerate(points_2d):
            if 0 <= x < w and 0 <= y < h:
                if binary_mask[y, x]:
                    selected_indices.append(i)

        if not selected_indices:
            print("投影方法未找到匹配的点")
            return None

        # 创建裁剪后的点云
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(self.points[selected_indices])

        if self.original_pcd.has_colors():
            colors = np.asarray(self.original_pcd.colors)
            cropped_pcd.colors = o3d.utility.Vector3dVector(colors[selected_indices])

        o3d.io.write_point_cloud(output_path, cropped_pcd)
        print(f"投影方法裁剪完成: {len(selected_indices)}/{self.point_count} 个点")

        return cropped_pcd

    def crop_with_spatial_filtering(self, output_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        使用掩码的空间信息进行粗略筛选
        """
        if self.original_pcd is None:
            raise ValueError("请先加载数据")

        print("=" * 60)
        print("方法: 空间筛选")
        print("=" * 60)

        # 找到前景像素的坐标
        mask_coords = np.column_stack(np.where(self.original_mask > 128))

        if len(mask_coords) == 0:
            print("掩码中没有前景区域")
            return None

        # 将点云坐标归一化（假设点云和图像有相似的分布）
        points_norm = (self.points - self.points.mean(axis=0)) / self.points.std(axis=0)

        # 粗略筛选：选择在掩码中心附近的点
        distances = np.linalg.norm(points_norm[:, :2], axis=1)
        threshold = np.percentile(distances, 70)  # 选择距离较小的70%的点

        selected_indices = np.where(distances < threshold)[0]

        # 创建裁剪后的点云
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(self.points[selected_indices])

        if self.original_pcd.has_colors():
            colors = np.asarray(self.original_pcd.colors)
            cropped_pcd.colors = o3d.utility.Vector3dVector(colors[selected_indices])

        o3d.io.write_point_cloud(output_path, cropped_pcd)
        print(f"空间筛选完成: {len(selected_indices)}/{self.point_count} 个点")

        return cropped_pcd

    def advanced_crop(self, output_path: str = "output_cropped.ply") -> Optional[o3d.geometry.PointCloud]:
        """
        高级方法：尝试多种匹配策略，选择最佳结果
        """
        print("=" * 60)
        print("高级点云裁剪策略")
        print("=" * 60)

        strategies = []

        # 策略1: 直接降采样匹配
        try:
            result_path = output_path.replace('.ply', '_downsampled.ply')
            cropped_pcd = self.crop_with_downsampling(result_path)
            if cropped_pcd:
                strategies.append(('降采样匹配', cropped_pcd))
        except Exception as e:
            print(f"降采样策略失败: {e}")

        # 策略2: 使用相机投影
        try:
            result_path = output_path.replace('.ply', '_projection.ply')
            cropped_pcd = self.crop_with_camera_projection(result_path)
            if cropped_pcd:
                strategies.append(('相机投影', cropped_pcd))
        except Exception as e:
            print(f"投影策略失败: {e}")

        # 策略3: 使用空间范围筛选
        try:
            result_path = output_path.replace('.ply', '_spatial.ply')
            cropped_pcd = self.crop_with_spatial_filtering(result_path)
            if cropped_pcd:
                strategies.append(('空间筛选', cropped_pcd))
        except Exception as e:
            print(f"空间筛选策略失败: {e}")

        # 选择最佳策略
        if strategies:
            print(f"\n找到 {len(strategies)} 种有效策略:")
            for i, (name, pcd) in enumerate(strategies):
                points_count = len(np.asarray(pcd.points))
                print(f"  {i + 1}. {name}: {points_count} 个点")

            # 选择点数最合适的策略
            best_name, best_pcd = min(strategies,
                                      key=lambda x: abs(len(np.asarray(x[1].points)) - 1000))
            print(f"\n选择最佳策略: {best_name}")

            # 保存最终结果
            o3d.io.write_point_cloud(output_path, best_pcd)
            self.cropped_pcd = best_pcd
            return best_pcd
        else:
            print("所有策略都失败了")
            return None

    def visualize_results(self, save_prefix: str = "result"):
        """
        可视化降采样和裁剪结果
        """
        if not hasattr(self, 'selected_indices'):
            print("无法可视化：请先运行裁剪方法")
            return

        # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原始掩码
        # axes[0, 0].imshow(self.original_mask, cmap='gray')
        # axes[0, 0].set_title(f'原始掩码\n{self.original_mask.shape[1]}x{self.original_mask.shape[0]}')
        # axes[0, 0].axis('off')
        #
        # # 2. 降采样后的掩码
        # if 'downsampled' in self.mask_info:
        #     axes[0, 1].imshow(self.mask_info['downsampled'], cmap='gray')
        #     axes[0, 1].set_title(
        #         f'降采样掩码\n{self.mask_info["downsampled"].shape[1]}x{self.mask_info["downsampled"].shape[0]}')
        # axes[0, 1].axis('off')
        #
        # # 3. 二值化掩码
        # if 'binary' in self.mask_info:
        #     axes[0, 2].imshow(self.mask_info['binary'], cmap='gray')
        #     axes[0, 2].set_title(f'二值化掩码\n前景像素: {np.sum(self.mask_info["binary"])}')
        # axes[0, 2].axis('off')

        # 4. 原始彩色图像（如果有）
        # if 'original' in self.image_info and self.image_info['original'] is not None:
        #     axes[1, 0].imshow(cv2.cvtColor(self.image_info['original'], cv2.COLOR_BGR2RGB))
        #     axes[1, 0].set_title(
        #         f'原始彩色图像\n{self.image_info["original"].shape[1]}x{self.image_info["original"].shape[0]}')
        # axes[1, 0].axis('off')
        #
        # # 5. 降采样彩色图像（如果有）
        # if 'downsampled' in self.image_info and self.image_info['downsampled'] is not None:
        #     axes[1, 1].imshow(cv2.cvtColor(self.image_info['downsampled'], cv2.COLOR_BGR2RGB))
        #     axes[1, 1].set_title(
        #         f'降采样彩色图像\n{self.image_info["downsampled"].shape[1]}x{self.image_info["downsampled"].shape[0]}')
        # axes[1, 1].axis('off')
        #
        # # 6. 点云分布可视化
        # axes[1, 2].scatter(self.points[:, 0], self.points[:, 1], s=1, alpha=0.3, label='所有点')
        # axes[1, 2].scatter(self.points[self.selected_indices, 0],
        #                    self.points[self.selected_indices, 1],
        #                    s=2, alpha=0.8, color='red', label='选中点')
        # axes[1, 2].set_title(f'点云分布\n选中: {len(self.selected_indices)}/{len(self.points)}')
        # axes[1, 2].legend()
        # axes[1, 2].set_aspect('equal')
        # axes[1, 2].grid(True)

        # plt.tight_layout()
        #
        # # 保存图像
        # plt.savefig(f"{save_prefix}_visualization.png", dpi=150, bbox_inches='tight')
        # plt.show()

        # 额外保存降采样后的掩码
        if 'binary' in self.mask_info:
            Image.fromarray((self.mask_info['binary'] * 255).astype(np.uint8)).save(
                f"{save_prefix}_downsampled_mask.png"
            )
            print(f"降采样后的掩码已保存: {save_prefix}_downsampled_mask.png")

    def display_point_clouds(self):
        """显示原始和裁剪后的点云"""
        if self.original_pcd:
            o3d.visualization.draw_geometries([self.original_pcd],
                                              window_name="原始点云")

        if self.cropped_pcd:
            o3d.visualization.draw_geometries([self.cropped_pcd],
                                              window_name="裁剪后的点云")

    def main(self, ply_data = None,
             mask_data = None,
             color_image = None,
             output_path = "r\yanpu_ur8\images\Mech\output_cropped.ply",
             show: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        主函数，直接接收数据对象而非路径。
        Args:
            ply_data: 点云数据，是open3d.geometry.PointCloud对象
            mask_data: 掩码数据，是 numpy 数组 (灰度图)
            color_image: 彩色图像 (BGR格式的 numpy 数组)，可选
            output_path: 裁剪后点云的保存路径
            show: 是否显示裁剪后的点云
        Returns:
            裁剪后的点云对象，失败返回 None
        """
        print("\n开始处理...")
        # 1. 加载点云
        self.original_pcd = ply_data
        self.points = np.asarray(self.original_pcd.points)
        self.point_count = len(self.points)
        print(f"点云点数: {self.point_count}")
        # 2. 加载掩码
        self.original_mask = np.array(mask_data)
        if len(self.original_mask.shape) > 2:
            self.original_mask = self.original_mask[:, :, 0]
        print(f"原始掩码尺寸: {self.original_mask.shape[1]}x{self.original_mask.shape[0]}")
        # 2. 加载彩色图像
        self.color_image = color_image
        if self.color_image is not None:
            self.color_image_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
            print(f"彩色图像尺寸: {self.color_image.shape[1]}x{self.color_image.shape[0]}")
        else:
            self.color_image = None
            print("未提供彩色图像")
        # 存储信息
        self.mask_info['original'] = self.original_mask
        self.image_info['original'] = self.color_image

        # 尝试降采样方法
        cropped_pcd = self.crop_with_downsampling(output_path)
        if cropped_pcd:
            print("\n降采样方法成功!")
        else:
            print("\n降采样方法失败，尝试高级方法...")
            cropped_pcd = self.advanced_crop(output_path)
            if not cropped_pcd:
                print("所有方法都失败了")
                return None

        if show and cropped_pcd is not None:
            o3d.visualization.draw_geometries([cropped_pcd],
                                              window_name="裁剪后的点云",
                                              width=640, height=480)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        return cropped_pcd


if __name__ == "__main__":
    # 裁剪点云
    # 点云路径
    ply_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur2\images\point_cloud_20260202_210153_482_repaired.ply"
    # 掩码路径
    mask_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur2\images\box_segmented_mask_20260202-205428.png"
    # 彩色图路径
    color_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur5\images\Mech\color.jpg"
    # 裁剪后的点云路径
    output_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur2\images\output_cropped2.ply"

    # 1. 加载点云
    original_pcd = o3d.io.read_point_cloud(ply_path)
    # 2. 加载掩码
    mask = Image.open(mask_path)
    # 3. 加载彩色图
    color_image = Image.open(color_path)

    ssss = PointCloudCropper()
    ssss.main(original_pcd, mask, color_image, output_path,show=True)





