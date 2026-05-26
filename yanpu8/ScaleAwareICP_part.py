import open3d as o3d
import numpy as np
import copy
import os
from sklearn.cluster import DBSCAN
import basis.robot_math as rm


class RotationOnlyICP:
    def __init__(self):
        self.cluster_center = None
        self.cluster_bbox_center = None
        self.fixed_translation = None
        self.initial_translation_only = None

    def debug_scales(self, ply=None, ply_path=None, stl_path=None):
        """详细调试尺度问题"""
        print("=" * 60)
        print("尺度调试信息")

        if ply is not None:
            scene = ply
        elif ply_path is not None:
            scene = o3d.io.read_point_cloud(ply_path)
        else:
            raise ValueError("必须提供 ply 或 ply_path")
        mesh = o3d.io.read_triangle_mesh(stl_path)

        scene_points = np.asarray(scene.points)
        model_vertices = np.asarray(mesh.vertices)

        # 场景分析
        scene_min = np.min(scene_points, axis=0)
        scene_max = np.max(scene_points, axis=0)
        scene_size = scene_max - scene_min
        scene_diag = np.linalg.norm(scene_size)

        # print("\n=== PLY场景分析 ===")
        # print(f"点数: {len(scene_points)}")
        # print(f"尺寸范围: [{scene_size[0]:.6f}, {scene_size[1]:.6f}, {scene_size[2]:.6f}]")
        # print(f"对角线长度: {scene_diag:.6f}")

        # 模型分析
        model_min = np.min(model_vertices, axis=0)
        model_max = np.max(model_vertices, axis=0)
        model_size = model_max - model_min
        model_diag = np.linalg.norm(model_size)

        # print("\n=== STL模型分析 ===")
        # print(f"顶点数: {len(model_vertices)}")
        # print(f"尺寸范围: [{model_size[0]:.6f}, {model_size[1]:.6f}, {model_size[2]:.6f}]")
        # print(f"对角线长度: {model_diag:.6f}")

        # 比例分析
        scale_ratio = scene_diag / model_diag if model_diag > 0 else 1.0
        # print(f"\n=== 尺度比例 ===")
        # print(f"场景/模型对角线比例: {scale_ratio:.6f}")

        return scene, mesh, scale_ratio

    def normalize_to_same_scale(self, scene, mesh, manual_scale=None):
        """将STL归一化到与PLY相同的尺度"""
        scene_points = np.asarray(scene.points)
        model_vertices = np.asarray(mesh.vertices)

        scene_diag = np.linalg.norm(
            np.max(scene_points, axis=0) - np.min(scene_points, axis=0))

        model_diag = np.linalg.norm(
            np.max(model_vertices, axis=0) - np.min(model_vertices, axis=0))

        if manual_scale is not None:
            scale_factor = manual_scale
        else:
            target_size = scene_diag * 0.3
            scale_factor = target_size / model_diag if model_diag > 0 else 1.0

        print(f"\n缩放因子: {scale_factor:.6f}")
        # print(f"模型原始对角线: {model_diag:.6f}")
        # print(f"缩放后对角线: {model_diag * scale_factor:.6f}")

        # 缩放模型
        scaled_vertices = model_vertices * scale_factor
        mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

        # 创建模型点云
        model_cloud = mesh.sample_points_uniformly(number_of_points=30000)

        return scene, model_cloud, scale_factor

    def find_main_cluster_bbox_center(self, points, eps=0.02, min_samples=30):
        """找到点云中最大聚类，并返回其包围盒中心"""
        print(f"\n进行DBSCAN聚类分析...")
        print(f"总点数: {len(points)}")

        # 执行DBSCAN聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        # 统计各聚类
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            print("未找到任何聚类，使用所有点的包围盒中心")
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            bbox_center = (min_pt + max_pt) / 2
            bbox = np.concatenate([min_pt, max_pt])

            self.cluster_center = np.mean(points, axis=0)
            self.cluster_bbox_center = bbox_center

            return bbox_center, points, 1.0, bbox

        # 找到最大的聚类
        main_cluster_idx = np.argmax(counts)
        main_cluster_label = unique_labels[main_cluster_idx]
        main_cluster_count = counts[main_cluster_idx]

        # 提取最大聚类的点
        main_cluster_mask = (labels == main_cluster_label)
        main_cluster_points = points[main_cluster_mask]

        # 计算聚类质心
        cluster_centroid = np.mean(main_cluster_points, axis=0)

        # 计算聚类包围盒
        cluster_min = np.min(main_cluster_points, axis=0)
        cluster_max = np.max(main_cluster_points, axis=0)
        cluster_size = cluster_max - cluster_min

        # 计算包围盒中心
        bbox_center = (cluster_min + cluster_max) / 2

        # 保存包围盒信息
        cluster_bbox = np.concatenate([cluster_min, cluster_max])
        cluster_ratio = main_cluster_count / len(points)

        # 保存到类属性中
        self.cluster_center = cluster_centroid
        self.cluster_bbox_center = bbox_center

        # print(f"聚类数量: {len(unique_labels)}")
        # print(f"最大聚类点数: {main_cluster_count}")
        # print(f"最大聚类占比: {cluster_ratio:.2%}")
        # print(f"聚类质心: {cluster_centroid}")
        # print(f"聚类包围盒中心: {bbox_center}")
        # print(f"聚类包围盒尺寸: {cluster_size}")

        return bbox_center, main_cluster_points, cluster_ratio, cluster_bbox

    def create_center_marker(self, center_point, size=0.02, color=[1, 0, 0]):
        """创建中心点标记"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.compute_vertex_normals()
        sphere.translate(center_point)
        sphere.paint_uniform_color(color)

        return sphere

    def create_bbox(self, min_point, max_point, color=[0, 1, 0], line_width=2):
        """创建包围盒的可视化"""
        points = [
            [min_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], max_point[2]],
            [min_point[0], max_point[1], max_point[2]]
        ]

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
        ]

        colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def visualize_cluster_with_bbox(self, scene, cluster_points, bbox_center, cluster_bbox):
        """可视化聚类及其包围盒"""
        print("\n" + "=" * 60)
        print("可视化聚类及其包围盒")
        print("=" * 60)

        # 复制场景
        scene_vis = copy.deepcopy(scene)
        scene_vis.paint_uniform_color([0.7, 0.7, 0.7])

        # 创建聚类点云
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_cloud.paint_uniform_color([1, 0, 0])

        # 创建包围盒
        min_point = cluster_bbox[:3]
        max_point = cluster_bbox[3:]
        bbox = self.create_bbox(min_point, max_point, color=[0, 1, 0])

        # 创建中心点标记
        centroid_marker = self.create_center_marker(
            self.cluster_center, size=0.015, color=[1, 1, 0]
        )

        bbox_center_marker = self.create_center_marker(
            bbox_center, size=0.02, color=[1, 0, 1]
        )

        # 几何体列表
        geometries = [scene_vis, cluster_cloud, bbox, centroid_marker, bbox_center_marker]

        # 坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)

        print(f"\n可视化说明:")
        print(f"  灰色点: 场景点云")
        print(f"  红色点: 主要聚类点")
        print(f"  绿色线框: 聚类包围盒")
        print(f"  黄色小球: 聚类质心")
        print(f"  紫色小球: 包围盒中心（用于对齐）")

        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="聚类分析（红色为聚类点，绿色为包围盒）",
            width=1024, height=768
        )

    def visualize_translation_only(self, scene, model, translation, bbox_center):
        """可视化只做平移的效果（不进行旋转）"""
        print("\n" + "=" * 60)
        print("可视化：只做平移的效果（使用包围盒中心）")
        print("=" * 60)

        # 复制几何体
        scene_vis = copy.deepcopy(scene)
        model_vis = copy.deepcopy(model)
        model_translated = copy.deepcopy(model)

        # 着色
        scene_vis.paint_uniform_color([0.7, 0.7, 0.7])
        model_vis.paint_uniform_color([1, 0, 0])
        model_translated.paint_uniform_color([0, 1, 0])

        # 创建只包含平移的变换矩阵
        T_translation_only = np.eye(4)
        T_translation_only[:3, 3] = translation
        self.initial_translation_only = T_translation_only

        # 应用平移变换
        model_translated.transform(T_translation_only)

        # 计算模型包围盒中心
        model_points = np.asarray(model_vis.points)
        model_min = np.min(model_points, axis=0)
        model_max = np.max(model_points, axis=0)
        model_bbox_center = (model_min + model_max) / 2

        # 计算平移后的模型包围盒中心
        model_translated_points = np.asarray(model_translated.points)
        model_translated_min = np.min(model_translated_points, axis=0)
        model_translated_max = np.max(model_translated_points, axis=0)
        model_translated_bbox_center = (model_translated_min + model_translated_max) / 2

        print(f"原始模型包围盒中心: {model_bbox_center}")
        print(f"平移后模型包围盒中心: {model_translated_bbox_center}")
        # print(f"目标聚类包围盒中心: {bbox_center}")
        # print(f"平移向量: {translation}")
        # print(f"平移距离: {np.linalg.norm(translation):.6f}")

        # 创建几何体列表
        geometries = [scene_vis, model_vis, model_translated]

        # 添加坐标轴
        coord_frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame_origin)

        # 原始模型包围盒中心（蓝色）
        orig_bbox_center_marker = self.create_center_marker(
            model_bbox_center, size=0.015, color=[0, 0, 1]
        )
        geometries.append(orig_bbox_center_marker)

        # 平移后模型包围盒中心（绿色）
        translated_bbox_center_marker = self.create_center_marker(
            model_translated_bbox_center, size=0.015, color=[0, 1, 0]
        )
        geometries.append(translated_bbox_center_marker)

        # 目标聚类包围盒中心（紫色）
        target_bbox_center_marker = self.create_center_marker(
            bbox_center, size=0.02, color=[1, 0, 1]
        )
        geometries.append(target_bbox_center_marker)

        # 聚类质心（黄色）- 用于对比
        if self.cluster_center is not None:
            centroid_marker = self.create_center_marker(
                self.cluster_center, size=0.015, color=[1, 1, 0]
            )
            geometries.append(centroid_marker)

        print(f"\n可视化说明:")
        print(f"  灰色点: 场景点云")
        print(f"  红色点: 原始模型（未变换）")
        print(f"  绿色点: 只做平移后的模型")
        print(f"  蓝色小球: 原始模型包围盒中心")
        print(f"  绿色小球: 平移后模型包围盒中心")
        print(f"  紫色小球: 目标聚类包围盒中心（对齐目标）")

        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="只做平移的效果（使用包围盒中心对齐）",
            width=1024, height=768
        )

        return model_translated

    def visualize_final_result(self, scene, model, transformation):
        """可视化最终配准结果"""
        print("\n" + "=" * 60)
        print("可视化最终配准结果")
        print("=" * 60)

        # 复制几何体
        scene_vis = copy.deepcopy(scene)
        model_original = copy.deepcopy(model)
        model_final = copy.deepcopy(model)

        # 着色
        scene_vis.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色 - 场景
        model_original.paint_uniform_color([1, 0, 0])  # 红色 - 原始模型
        model_final.paint_uniform_color([0, 0, 1])  # 蓝色 - 最终对齐模型

        # 应用最终变换
        model_final.transform(transformation)

        # 创建几何体列表
        geometries = [scene_vis, model_original, model_final]

        # 添加坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)

        # 显示聚类包围盒中心（如果存在）
        if self.cluster_bbox_center is not None:
            bbox_center_marker = self.create_center_marker(
                self.cluster_bbox_center, size=0.02, color=[1, 0, 1]
            )
            geometries.append(bbox_center_marker)

            # 在包围盒中心处添加小坐标轴
            bbox_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=self.cluster_bbox_center
            )
            geometries.append(bbox_coord)

        print(f"\n可视化说明:")
        print(f"  灰色点: 场景点云")
        print(f"  红色点: 原始模型（未变换）")
        print(f"  蓝色点: 最终对齐模型（旋转优化后）")
        if self.cluster_bbox_center is not None:
            print(f"  紫色小球: 聚类包围盒中心（对齐目标）")

        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="最终配准结果",
            width=1024, height=768
        )

    def visualize_all_stages(self, scene, model, transformation):
        """可视化所有阶段的结果对比"""
        print("\n" + "=" * 60)
        print("可视化所有阶段对比")
        print("=" * 60)

        # 复制几何体
        scene_vis = copy.deepcopy(scene)
        model_original = copy.deepcopy(model)
        model_translated_only = copy.deepcopy(model)
        model_final = copy.deepcopy(model)

        # 着色
        scene_vis.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色 - 场景
        model_original.paint_uniform_color([1, 0, 0])  # 红色 - 原始模型
        model_translated_only.paint_uniform_color([0, 1, 0])  # 绿色 - 只平移
        model_final.paint_uniform_color([0, 0, 1])  # 蓝色 - 最终结果

        # 应用变换
        if self.initial_translation_only is not None:
            model_translated_only.transform(self.initial_translation_only)

        model_final.transform(transformation)

        # 几何体列表
        geometries = [scene_vis, model_original, model_translated_only, model_final]

        # 坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)

        # 显示聚类包围盒中心
        if self.cluster_bbox_center is not None:
            bbox_center_marker = self.create_center_marker(
                self.cluster_bbox_center, size=0.02, color=[1, 0, 1]
            )
            geometries.append(bbox_center_marker)

        print(f"\n可视化说明:")
        print(f"  灰色点: 场景点云")
        print(f"  红色点: 原始模型（未变换）")
        print(f"  绿色点: 只做平移后的模型（无旋转）")
        print(f"  蓝色点: 旋转优化后的最终模型")
        if self.cluster_bbox_center is not None:
            print(f"  紫色小球: 聚类包围盒中心")

        print(f"\n配准流程:")
        print(f"  1. 原始模型（红色）→ 只平移（绿色）")
        print(f"  2. 只平移（绿色）→ 旋转优化（蓝色）")

        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="所有阶段对比：原始/只平移/最终结果",
            width=1024, height=768
        )

    def create_translation_arrow(self, start_point, end_point, color=[1, 0, 0]):
        """创建表示平移方向的箭头"""
        try:
            # 计算箭头方向
            direction = end_point - start_point
            distance = np.linalg.norm(direction)

            if distance < 1e-6:
                return None

            # 归一化方向向量
            direction = direction / distance

            # 创建箭头（圆柱 + 圆锥）
            arrow_length = distance * 0.8
            cone_height = distance * 0.2

            # 创建圆柱（箭头主体）
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.005,
                height=arrow_length
            )
            cylinder.paint_uniform_color(color)

            # 创建圆锥（箭头头部）
            cone = o3d.geometry.TriangleMesh.create_cone(
                radius=0.01,
                height=cone_height
            )
            cone.paint_uniform_color(color)

            # 计算旋转使箭头指向正确方向
            default_direction = np.array([0, 0, 1])
            rotation_axis = np.cross(default_direction, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_angle = np.arccos(np.dot(default_direction, direction))
                rotation_matrix = self.rotation_matrix_from_axis_angle(
                    rotation_axis / rotation_axis_norm, rotation_angle
                )

                cylinder.rotate(rotation_matrix, center=[0, 0, 0])
                cone.rotate(rotation_matrix, center=[0, 0, 0])

            # 计算位置
            mid_point = start_point + direction * (arrow_length / 2)
            cylinder.translate(mid_point)

            cone_center = start_point + direction * (arrow_length + cone_height / 2)
            cone.translate(cone_center)

            # 合并箭头
            arrow = cylinder + cone
            return arrow

        except Exception as e:
            print(f"创建箭头失败: {e}")
            return None

    def rotation_matrix_from_axis_angle(self, axis, angle):
        """根据轴角创建旋转矩阵"""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    # 用于U625的匹配
    def simple_icp_rotation_only(self, scene, model, max_distance=0.02,
                                 use_cluster_center=True, eps=0.02, min_samples=30,
                                 use_bbox_center=True, show = False):
        """执行只优化旋转的ICP，用于U625的匹配"""

        print("\n" + "=" * 60)
        print("开始只优化旋转的ICP配准")
        if use_bbox_center:
            print("使用聚类包围盒中心进行初始对齐")
        else:
            print("使用聚类质心进行初始对齐")
        print("=" * 60)

        # 步骤1：下采样
        scene_down = scene.voxel_down_sample(0.005)
        model_down = model.voxel_down_sample(0.005)

        print(f"下采样后:")
        print(f"  场景点数: {len(scene_down.points)}")
        print(f"  模型点数: {len(model_down.points)}")

        # 步骤2：获取点数据
        scene_points = np.asarray(scene_down.points)
        model_points = np.asarray(model_down.points)

        # 步骤3：计算模型包围盒中心
        model_min = np.min(model_points, axis=0)
        model_max = np.max(model_points, axis=0)
        model_bbox_center = (model_min + model_max) / 2
        print(f"模型包围盒中心: {model_bbox_center}")

        # 步骤4：选择场景中心点
        if use_cluster_center:
            # 使用聚类包围盒中心或质心
            bbox_center, main_cluster_points, cluster_ratio, cluster_bbox = self.find_main_cluster_bbox_center(
                scene_points, eps=eps, min_samples=min_samples
            )

            if use_bbox_center:
                scene_center = bbox_center
                print(f"使用聚类包围盒中心进行初始对齐")
                # print(f"聚类包围盒中心: {scene_center}")
            else:
                scene_center = self.cluster_center
                print(f"使用聚类质心进行初始对齐")
                # print(f"聚类质心: {scene_center}")

            # 可视化聚类和包围盒
            if show:
                self.visualize_cluster_with_bbox(
                    scene_down, main_cluster_points, bbox_center, cluster_bbox
                )
        else:
            # 使用整个场景的包围盒中心
            scene_min = np.min(scene_points, axis=0)
            scene_max = np.max(scene_points, axis=0)
            scene_center = (scene_min + scene_max) / 2
            print(f"使用整个场景包围盒中心进行初始对齐")
            # print(f"场景包围盒中心: {scene_center}")

        # 步骤5：计算初始平移
        initial_translation = scene_center - model_bbox_center
        self.fixed_translation = initial_translation

        # print(f"\n初始平移向量（固定）: {initial_translation}")
        # print(f"平移距离: {np.linalg.norm(initial_translation):.6f}")

        # 步骤5.5：可视化只做平移的效果
        if show:
            print("\n正在显示只做平移的效果...")
            self.visualize_translation_only(
                scene_down, model_down, initial_translation, scene_center
            )

        # 步骤6：执行只优化旋转的ICP
        transformation, fitness, rmse = self.rotation_only_icp_kabsch(
            model_down, scene_down, initial_translation, max_distance
        )

        # print(f"\n" + "=" * 60)
        # print("旋转优化完成")
        # print("=" * 60)
        print(f"最终匹配度: {fitness:.4f}")
        print(f"最终RMSE: {rmse:.6f}")

        return transformation

    # 用于U625的匹配
    def rotation_only_icp_kabsch(self, source, target, initial_translation, max_distance=0.02, max_iterations=50):
        """使用Kabsch算法只优化旋转"""

        print("\n" + "=" * 60)
        print("开始旋转优化（Kabsch算法）")

        # 复制点云
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # 获取点数据
        source_points = np.asarray(source_temp.points)
        target_points = np.asarray(target_temp.points)

        # 构建KD树用于快速最近邻搜索
        target_kdtree = o3d.geometry.KDTreeFlann(target_temp)

        # 初始变换：只应用平移（不旋转）
        T_current = np.eye(4)
        T_current[:3, 3] = initial_translation

        # 将源点云平移到初始位置
        source_temp.transform(T_current)
        source_points = np.asarray(source_temp.points)

        # 使用包围盒中心作为旋转中心
        rotation_center = self.cluster_bbox_center if self.cluster_bbox_center is not None else np.mean(target_points,
                                                                                                        axis=0)
        # print(f"旋转中心: {rotation_center}")
        # print(f"固定平移: {initial_translation}")

        best_fitness = 0
        best_transformation = T_current.copy()
        convergence_count = 0
        current_rmse = 0

        for iteration in range(max_iterations):
            # print(f"\n--- 迭代 {iteration + 1}/{max_iterations} ---")

            # 建立点对应关系
            correspondences = []
            distances = []

            for i, src_point in enumerate(source_points):
                [k, idx, _] = target_kdtree.search_knn_vector_3d(src_point, 1)

                if k > 0:
                    target_point = target_points[idx[0]]
                    distance = np.linalg.norm(src_point - target_point)

                    if distance < max_distance:
                        correspondences.append((i, idx[0]))
                        distances.append(distance)

            # if len(correspondences) < 1:
            #     print(f"警告：只有{len(correspondences)}个有效点对，停止迭代")

            # 提取对应点
            src_corr_indices = [c[0] for c in correspondences]
            tgt_corr_indices = [c[1] for c in correspondences]

            src_corr_points = source_points[src_corr_indices]
            tgt_corr_points = target_points[tgt_corr_indices]

            # 使用Kabsch算法计算最优旋转
            src_centered = src_corr_points - rotation_center
            tgt_centered = tgt_corr_points - rotation_center

            H = src_centered.T @ tgt_centered

            U, S, Vt = np.linalg.svd(H)

            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            # print(f"旋转矩阵行列式: {np.linalg.det(R):.6f}")

            # 提取旋转角度
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0

            # angles_deg = np.degrees([x, y, z])
            # print(f"旋转角度(度): [{angles_deg[0]:.2f}, {angles_deg[1]:.2f}, {angles_deg[2]:.2f}]")

            # 应用旋转
            T_current[:3, :3] = R @ T_current[:3, :3]

            # 重新计算源点云位置
            source_temp = copy.deepcopy(source)
            source_temp.rotate(R, center=rotation_center)
            source_temp.translate(initial_translation)

            source_points = np.asarray(source_temp.points)

            # 计算评估指标
            current_fitness = len(correspondences) / len(source_points)
            current_rmse = np.sqrt(np.mean(np.square(distances))) if distances else 0

            # print(f"有效点对: {len(correspondences)}/{len(source_points)}")
            # print(f"匹配度: {current_fitness:.4f}")
            # print(f"RMSE: {current_rmse:.6f}")

            # 检查收敛
            if iteration > 0:
                fitness_improvement = current_fitness - best_fitness

                if abs(fitness_improvement) < 1e-6:
                    convergence_count += 1
                    # print(f"收敛计数器: {convergence_count}/3")
                else:
                    convergence_count = 0

                if convergence_count >= 3:
                    # print(f"\n✓ 旋转优化已收敛（迭代{iteration + 1}次）")
                    # print(f"最终匹配度: {best_fitness:.4f}")
                    break

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_transformation = T_current.copy()

            # if iteration == max_iterations - 1:
            #     print(f"\n⚠ 达到最大迭代次数{max_iterations}")
            #     print(f"最佳匹配度: {best_fitness:.4f}")

        return best_transformation, best_fitness, current_rmse

    # 用于U625的匹配
    def run_rotation_only_icp(self, ply, ply_path, stl_path, manual_scale=None,
                              use_cluster_center=True, use_bbox_center=True,
                              eps=0.02, min_samples=30, show = False):
        """运行只优化旋转的ICP配准,用于U625的匹配"""

        print("=" * 60)
        print("只优化旋转的ICP配准")
        print("=" * 60)

        # 1. 调试尺度
        scene, mesh, auto_scale = self.debug_scales(ply, ply_path, stl_path)

        # 2. 创建原始模型点云
        original_model_cloud = mesh.sample_points_uniformly(number_of_points=20000)

        # 3. 归一化尺度
        scene, scaled_model_cloud, scale_factor = self.normalize_to_same_scale(
            scene, mesh, manual_scale)

        # 4. 执行只优化旋转的ICP
        transformation = self.simple_icp_rotation_only(
            scene, scaled_model_cloud,
            use_cluster_center=use_cluster_center,
            use_bbox_center=use_bbox_center,
            eps=eps,
            min_samples=min_samples,
            show=show
        )

        # 5. 可视化最终结果
        if show:
            # 5.1 显示最终配准结果
            self.visualize_final_result(scene, scaled_model_cloud, transformation)

            # 5.2 显示所有阶段对比
            self.visualize_all_stages(scene, scaled_model_cloud, transformation)

        # 6. 保存结果
        output_dir = "rotation_only_results"
        os.makedirs(output_dir, exist_ok=True)

        # 保存变换矩阵
        np.savetxt(os.path.join(output_dir, "rotation_only_transformation.txt"), transformation)

        if self.initial_translation_only is not None:
            np.savetxt(os.path.join(output_dir, "translation_only_matrix.txt"),
                       self.initial_translation_only)

        # 保存详细信息
        with open(os.path.join(output_dir, "rotation_only_info.txt"), 'w') as f:
            f.write(f"=== 只优化旋转的ICP配准结果 ===\n\n")
            f.write(f"对齐方式: {'聚类包围盒中心对齐' if use_bbox_center else '聚类质心对齐'}\n")
            f.write(f"固定平移向量: {self.fixed_translation}\n")
            f.write(f"平移距离: {np.linalg.norm(self.fixed_translation):.6f}\n")
            f.write(f"使用缩放因子: {scale_factor:.6f}\n")
            f.write(f"使用聚类中心初始化: {use_cluster_center}\n")
            f.write(f"使用包围盒中心: {use_bbox_center}\n")

            if self.cluster_center is not None:
                f.write(f"聚类质心: {self.cluster_center}\n")
            if self.cluster_bbox_center is not None:
                f.write(f"聚类包围盒中心: {self.cluster_bbox_center}\n")

            # 提取旋转矩阵和欧拉角
            R = transformation[:3, :3]
            f.write(f"\n最终旋转矩阵:\n")
            for i in range(3):
                f.write(f"  [{R[i, 0]:.6f}, {R[i, 1]:.6f}, {R[i, 2]:.6f}]\n")

            # 计算欧拉角
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0

            angles_deg = np.degrees([x, y, z])
            f.write(f"\n欧拉角(度): [{angles_deg[0]:.2f}, {angles_deg[1]:.2f}, {angles_deg[2]:.2f}]\n")

        print(f"\n结果已保存到: {output_dir}")
        print(f"固定平移向量: {self.fixed_translation}")
        print(f"对齐方式: {'聚类包围盒中心' if use_bbox_center else '聚类质心'}")

        return transformation, scale_factor

    # 用于U1的平移匹配
    def simple_icp_with_scale_fix(self, ply, ply_path, stl_path, manual_scale=None,
                                use_cluster_center=True, use_bbox_center=True, eps=0.02, min_samples=30,
                                max_distance = 0.02,
                                show = False):
        """
        计算平移和旋转对齐，进行迭代。用于U1的平移匹配
        输入：
            ply:点云文件
            ply_path:点云文件路径
            stl_path:模型路径
            manual_scale:是否自动尺度修正
            use_cluster_center:是否使用聚类中心确定初始位置
            use_bbox_center:是否使用聚类包围盒中心
            eps:DBSCAN 聚类半径，控制点云邻域距离
            min_samples:DBSCAN 最小点数，决定核心点阈值
            max_distance:ICP中判断对应点对是否为内点的距离阈值。两点距离小于该阈值才被认为是一对有效匹配，用于迭代求解变换。
            show:是否展示对齐结果
        返回：
            translation:平移矩阵
        """
        # 1. 调试尺度
        scene, mesh, auto_scale = self.debug_scales(ply, ply_path, stl_path)
        # 2. 归一化尺度
        scene, scaled_model_cloud, scale_factor = self.normalize_to_same_scale(
            scene, mesh, manual_scale)
        # 3.下采样，大幅减少点云数量，降低计算开销，同时过滤噪声
        scene_down = scene.voxel_down_sample(0.005)
        model_down = scaled_model_cloud.voxel_down_sample(0.005)
        # 4.计算法线,目标点云（场景）的法线信息，用于后面精确配准
        scene_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        model_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        # 5.计算模型包围盒中心
        scene_points = np.asarray(scene_down.points)
        model_points = np.asarray(model_down.points)
        model_min = np.min(model_points, axis=0)
        model_max = np.max(model_points, axis=0)
        model_bbox_center = (model_min + model_max) / 2
        # print(f"模型包围盒中心: {model_bbox_center}")
        # 6.计算场景中心点
        if use_cluster_center:  # 使用聚类中心
            # 使用DBSCAN找到主要聚类，并计算聚类质心
            bbox_center, main_cluster_points, cluster_ratio, cluster_bbox = self.find_main_cluster_bbox_center(
                scene_points, eps=eps, min_samples=min_samples
            )
            if use_bbox_center: # 使用聚类包围盒中心
                scene_center = bbox_center
                # print(f"使用聚类包围盒中心: {scene_center}")
            else:
                scene_center = self.cluster_center  # 聚类质心
                # print(f"使用聚类质心: {scene_center}")
        else:   # 不使用聚类中心，使用全局包围盒中心
            scene_min = np.min(scene_points, axis=0)
            scene_max = np.max(scene_points, axis=0)
            scene_center = (scene_min + scene_max) / 2
            # print(f"使用全局包围盒中心: {scene_center}")

        # 7.计算初始平移向量
        translation = scene_center - model_bbox_center
        # print(f"平移向量: {translation}")
        # 8.使用点-面ICP进行精确配准，同时优化旋转和平移
        T_init = np.eye(4)
        T_init[:3, 3] = translation # 初始的姿态定为无旋转
        # 还可以设置初始姿态

        result = o3d.pipelines.registration.registration_icp(
            model_down, # 待变换的点云
            scene_down, # 目标点云
            max_distance,   # 最大对应距离阈值，超过该距离的对应点对被忽略。
            T_init, # 初始变换矩阵
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # 计算点到平面误差
            o3d.pipelines.registration.ICPConvergenceCriteria(  # 迭代终止条件
                max_iteration=200,  # 最大迭代次数
                relative_fitness=1e-6,  # 连续两次迭代之间适应度
                relative_rmse=1e-6) # 均方根误差的相对变化
        )
        print(f"\nICP结果:")
        print(f"  匹配度: {result.fitness:.4f}")   # 范围 [0,1]，越大越好
        print(f"  误差: {result.inlier_rmse:.6f}")    # 越小越好
        transformation = result.transformation  # 齐次变换矩阵
        translation = transformation[:3,3]  # 平移向量
        # 9.可视化结果
        if show:
            self.visualize_final_result(scene, scaled_model_cloud, transformation)

        return translation
# 使用示例
if __name__ == "__main__":
    PLY_PATH = r"C:\Users\cc\Documents\Github\cc-wrs\chenchen\yanpu_ur\output_cropped.ply"
    STL_PATH = r"C:\Users\cc\Documents\Github\cc-wrs\chenchen\yanpu_ur\object\U625.STL"
    ply = None

    # 创建实例并运行只优化旋转的ICP
    icp = RotationOnlyICP()

    # 使用包围盒中心对齐（推荐）
    transformation, scale_factor = icp.run_rotation_only_icp(
        ply, PLY_PATH, STL_PATH,
        manual_scale=1,
        use_cluster_center=True,
        use_bbox_center=True,
        eps=0.02,
        min_samples=30,
        show = True
    )

    print(f"\n最终变换矩阵:")
    print(transformation)