from array import array

import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from typing import Literal

from numpy import float32
from ultralytics import YOLO

import basis.robot_math as rm


class ObjectDetector:  # 目标检测
    def __init__(self, aruco_dict=None, aruco_params=None,
                 obj_type: Literal['gasket', 'blot', 'nut'] = 'gasket'):  # 初始化
        marker_0_pos = np.array([0, 0])
        marker_1_pos = marker_0_pos + np.array([0.23, 0])
        marker_2_pos = marker_0_pos + np.array([0, 0.17])
        marker_3_pos = marker_0_pos + np.array([0.23, 0.17])
        self.marker_real_coords = {0: marker_0_pos, 1: marker_1_pos, 2: marker_2_pos, 3: marker_3_pos}  # 四个真实点坐标
        self.detect_obj_name = obj_type  # 目标种类
        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_250)  # ？
        self.aruco_params = aruco_params or aruco.DetectorParameters()  # ？
        yolo_pth_path = f'./yolo_pths/{self.detect_obj_name}/best.pt'  # yolo路径
        self.yolo_model = YOLO(yolo_pth_path)  # 加载yolo模型

    def detect_aruco_pixels(self, image, draw=True):  # 检测ArUco标记

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            print("未检测到任何 ArUco 标记")
            return None

        ids = ids.flatten()
        marker_px = {}
        for i, marker_id in enumerate(ids):
            if marker_id not in [0, 1, 2, 3]:
                continue  # 排除非目标ID

            center = np.mean(corners[i][0], axis=0)
            marker_px[marker_id] = center

            if draw:
                cv2.circle(image, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                cv2.putText(image, str(marker_id), tuple(center.astype(int) + np.array([5, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(marker_px) < 4:
            detected_ids = sorted(marker_px.keys())  # 已检测到的 ID
            print(f"检测到{len(marker_px)}个码：{detected_ids}，未齐 4 个（需要 ID 0–3）")
            cv2.imshow("wrong image", image)
            cv2.waitKey(0)
            return None

        aruco.drawDetectedMarkers(image, corners, ids)  # 画出检测到的标记的边界和ID

        return marker_px, image  # marker_px为二维码中心坐标字典，image为标注中心点后图像

    @staticmethod  # 把函数放进类中，给代码“贴标签”，让功能聚类、命名清晰，说明后面的函数都与目标检测有关
    def warp_to_marker_frame(image, aruco_px_dict, output_size=(460, 340)):  # 透视变换（像素→标准平面），把梯形/四边形拉成矩形，消除拍摄角度造成的变形
        """
        使用四个 ArUco 标记进行透视变换，将 marker 围成的区域变为矩形。
        默认顺序为：
            ID 0 - 左上
            ID 1 - 右上
            ID 2 - 左下
            ID 3 - 右下
        """
        try:
            p0 = np.array(aruco_px_dict[0], dtype=np.float32)  # 左上
            p1 = np.array(aruco_px_dict[1], dtype=np.float32)  # 右上
            p2 = np.array(aruco_px_dict[2], dtype=np.float32)  # 左下
            p3 = np.array(aruco_px_dict[3], dtype=np.float32)  # 右下
        except KeyError:
            print("marker ID 不完整，无法进行透视变换")
            return None, None

        # 构造目标矩形区域
        src_pts = np.array([p0, p1, p2, p3], dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [0, output_size[1] - 1],
            [output_size[0] - 1, output_size[1] - 1]
        ], dtype=np.float32)

        # 仿射变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, output_size)

        return warped, M

    def detect_objs_by_yolo(self, image):  # YOLO检测零件
        test_image = image
        results = self.yolo_model(test_image, conf=0.35)

        for result in results:
            xywhr = result.obb.xywhr  # 中心点 (x,y) 宽高 (w, h) 角度 r(弧度制)
            xyxyxyxy = result.obb.xyxyxyxy  # x1 y1 x2 y2 x3 y3 x4 y4
            corners = xyxyxyxy.cpu().numpy().reshape((-1, 4, 2))  # shape: (4, 2)
            centers = xywhr.cpu().numpy()[:, :2]
            classes = [result.names[cls.item()] for cls in result.obb.cls.int()]
            result_img = result.plot()
            centers.tolist()

            return result_img, corners, centers, classes

    def pixel_to_real(self, px, output_size=(460, 340)):  # 像素→真实坐标转换函数
        """在仿射图中计算像素点相对 ID0 的实际物理坐标"""
        dx_real = np.linalg.norm(self.marker_real_coords[1] - self.marker_real_coords[0])  # 通常是 0.23m
        dy_real = np.linalg.norm(self.marker_real_coords[2] - self.marker_real_coords[0])  # 通常是 0.17m
        w, h = output_size

        # 将像素转换为相对于 ID0 的物理坐标
        scale_x = dx_real / w  # x方向比例系数
        scale_y = dy_real / h  # y方向比例系数
        image_coord_x = px[0] * scale_x
        image_coord_y = px[1] * scale_y
        image_coord_z = 0
        image_coord = np.array([image_coord_x, image_coord_y, image_coord_z])
        convert_pos = np.array([0.7125, -0.0875, 0.033]) + np.array([0.17, 0, 0])
        # convert_pos = np.array([0.668, -0.182, 0]) + np.array([0.215, 0.095, 0.018])
        convert_rot = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-np.pi / 2)  # 把“ArUco 坐标系”通过旋转变换转成“机器人基坐标系”
        return image_coord.dot(convert_rot) + convert_pos  # 平移变换

    @staticmethod
    def plot_result(original_image, result_image, aruco_px_dict):  # 显示左侧原始图像和右侧透视变换图，并标注 ArUco markers 和 dataset 检测结果
        """
        显示左侧原始图像和右侧透视变换图，并标注 ArUco markers 和 dataset 检测结果
        """

        # 在原图中画 ArUco markers
        orig_img_vis = original_image.copy()
        for marker_id, pt in aruco_px_dict.items():
            pt_int = tuple(pt.astype(int))
            cv2.circle(orig_img_vis, pt_int, 5, (0, 0, 255), -1)
            cv2.circle(orig_img_vis, pt_int, 5, (0, 0, 255), -1)
            cv2.putText(orig_img_vis, f"id:{marker_id}", pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 连线：默认连 id0-id1（x方向），id0-id2（y方向）
        if all(k in aruco_px_dict for k in [0, 1, 2]):
            cv2.line(orig_img_vis, tuple(aruco_px_dict[0].astype(int)), tuple(aruco_px_dict[1].astype(int)),
                     (0, 255, 0), 2)
            cv2.line(orig_img_vis, tuple(aruco_px_dict[0].astype(int)), tuple(aruco_px_dict[2].astype(int)),
                     (0, 255, 0), 2)

        # --- 绘图 ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 左图：原图 + ArUco
        axs[0].imshow(cv2.cvtColor(orig_img_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image with ArUco Markers")

        # 右图：变换图 + dataset
        axs[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Detecting result by yolo v11")
        axs[1].set_xlim([0, result_image.shape[1]])
        axs[1].set_ylim([result_image.shape[0], 0])  # 保持左上为原点

        plt.tight_layout()
        plt.show()

    def get_blot_head_pos(self, image, pts, pixel_to_real_func=True, show=True):  # 检测螺钉头位置
        """
        可视化螺钉头部判断逻辑。
        参数：
            image: 输入图像（BGR 或灰度）
            pts: 矩形四个点，形状 (4, 2)
            pixel_to_real_func: 可选转换函数（像素->真实坐标）
            show: 是否显示 matplotlib 图像
        返回：
            blot_head_pos: 选择的点（真实坐标或像素坐标）
        """
        image_vis = image.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        pts = np.array(pts, dtype=np.float32).reshape(4, 2)

        def edge_length(pt1, pt2):
            return np.linalg.norm(pt1 - pt2)

        def pts_center(pt1, pt2):
            return np.mean(np.array([pt1, pt2]), axis=0)

        def get_mean_gray(rect_pts):
            mask = np.zeros_like(gray, dtype=np.uint8)
            rect_pts = rect_pts.astype(np.int32)
            cv2.fillPoly(mask, [rect_pts], 255)     # 不用管警告
            mean_val = cv2.mean(gray, mask=mask)[0]
            return mean_val

        edge1 = edge_length(pts[0], pts[1])
        edge2 = edge_length(pts[1], pts[2])

        if edge1 > edge2:  # 横向
            lft_region = np.array([pts[0], pts_center(pts[0], pts[1]), pts_center(pts[2], pts[3]), pts[3]])
            rgt_region = np.array([pts_center(pts[0], pts[1]), pts[1], pts[2], pts_center(pts[2], pts[3])])
            lft_mean = get_mean_gray(lft_region)
            rgt_mean = get_mean_gray(rgt_region)
            selected = pts_center(pts[0], pts[3]) if lft_mean > rgt_mean else pts_center(pts[1], pts[2])
            target = pts_center(pts[1], pts[2]) if lft_mean > rgt_mean else pts_center(pts[0], pts[3])
        else:  # 竖向
            lft_region = np.array([pts[1], pts_center(pts[1], pts[2]), pts_center(pts[0], pts[3]), pts[0]])
            rgt_region = np.array([pts_center(pts[1], pts[2]), pts[2], pts[3], pts_center(pts[0], pts[3])])
            lft_mean = get_mean_gray(lft_region)
            rgt_mean = get_mean_gray(rgt_region)
            selected = pts_center(pts[0], pts[1]) if lft_mean > rgt_mean else pts_center(pts[2], pts[3])
            target = pts_center(pts[2], pts[3]) if lft_mean > rgt_mean else pts_center(pts[0], pts[1])

        # 画图
        def draw_polygon(img, pts, color, label=None):
            pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=2)
            if label:
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(img, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        draw_polygon(image_vis, pts, (255, 0, 0), "bbox")
        draw_polygon(image_vis, lft_region, (0, 255, 0), "left")
        draw_polygon(image_vis, rgt_region, (0, 255, 255), "right")
        cv2.circle(image_vis, tuple(selected.astype(int)), 6, (0, 0, 255), -1)
        cv2.putText(image_vis, "selected", tuple(selected.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        if show:
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
            plt.title("Blot Head Region Detection")
            plt.axis('off')
            plt.show()

        if pixel_to_real_func:
            return self.pixel_to_real(selected), self.pixel_to_real(target)
        else:
            return selected, target

    def get_vec_along_shot_edge(self, corners):  # 输出手爪夹持方向（短边）
        d01 = np.linalg.norm(corners[0] - corners[1])
        d12 = np.linalg.norm(corners[1] - corners[2])
        if d01 > d12:
            long_vec = self.pixel_to_real(corners[1]) - self.pixel_to_real(corners[0])
        else:
            long_vec = self.pixel_to_real(corners[2]) - self.pixel_to_real(corners[1])
        perp_vec = np.array([-long_vec[1], long_vec[0]])
        perp_unit = perp_vec / np.linalg.norm(perp_vec)
        return np.hstack([perp_unit, 0.0])

    def get_center_and_vec_in_real(self, image, centers, corners, classes):  # 把2D像素坐标，变成机器人基坐标系下的3D抓取点+抓取方向向量
        center_list = []
        vec_list = []
        if self.detect_obj_name == 'gasket':  # 垫圈
            for i, class_name in enumerate(classes):
                center_list.append(self.pixel_to_real(centers[i]))
                if class_name == 'gasketstack':
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))
                else:
                    vec_list.append(np.array([0, 1, 0]))
        elif self.detect_obj_name == 'blot':  # 螺栓
            for i, class_name in enumerate(classes):
                if class_name == 'single':
                    blot_head, blot_tail = self.get_blot_head_pos(image, corners[i], show=False) + np.array(
                        [0, 0, 0.014])
                    grasp_center = blot_head
                    grasp_vec = rm.unit_vector(blot_tail - blot_head)
                elif class_name == 'stack':
                    grasp_center = self.pixel_to_real(centers[i]) + np.array([0, 0, 0.017])
                    grasp_vec = self.get_vec_along_shot_edge(corners[i])
                elif class_name == 'single_standing':
                    grasp_center = self.pixel_to_real(centers[i]) + np.array([0, 0, 0.022])
                    grasp_vec = np.array([0, 1, 0])
                else:
                    raise Exception('Unknown class: {}'.format(class_name))
                center_list.append(grasp_center)
                vec_list.append(grasp_vec)
        elif self.detect_obj_name == 'nut':  # 螺母
            for i, class_name in enumerate(classes):
                if class_name == 'stack':
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, 0.01]))
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))
                elif class_name == 'single_standing':
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, 0.018]))
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))
                else:
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, 0.01]))
                    vec_list.append(np.array([1, 0, 0]))

        return center_list, vec_list

    def run_on_image(self, image, draw):  # 输出最终分类
        aruco_button =False  # 是否用aruco检测
        if aruco_button:  # 使用aruco检测
            aruco_px, img_marked = self.detect_aruco_pixels(image, draw=draw)  # 检测二维码坐标
            if not aruco_px or not all(k in aruco_px for k in [0, 1, 2, 3]):
                print("缺少构建坐标系所需的 ArUco 标记 (0,1,2,3)")
                return []
            else:
                print("二维码位置:", aruco_px)
        else:  # 不使用aruco检测
            aruco_px = {3: np.array([603, 59.25], dtype=np.float32),
                        2: np.array([116.75, 71.5], dtype=np.float32),
                        0: np.array([130, 431.5], dtype=np.float32),
                        1: np.array([610.75, 414.75], dtype=np.float32)}

        warped_image, M = self.warp_to_marker_frame(image, aruco_px, output_size=(460, 340))  # 透视矫正，把平行四边形变为正方形
        result_img, obj_corners, obj_centers, obj_classes = self.detect_objs_by_yolo(warped_image)  # 目标检测
        self.plot_result(image, result_img, aruco_px)  # 画出结果
        obj_center_list, obj_vec_list = self.get_center_and_vec_in_real(warped_image, obj_centers, obj_corners,
                                                                        obj_classes)
        single_obj, standing_obj, stack_obj = [], [], []
        for i, obj_class in enumerate(obj_classes):
            if obj_class.lower() in ['single', 'gasketsingle']:
                single_obj.append([obj_center_list[i], obj_vec_list[i]])
            elif obj_class.lower() in ['stack', 'gasketstack']:
                stack_obj.append([obj_center_list[i], obj_vec_list[i]])
            elif obj_class.lower() == 'single_standing':
                standing_obj.append([obj_center_list[i], obj_vec_list[i]])

        return single_obj, stack_obj, standing_obj


if __name__ == "__main__":
    def main():
        image = cv2.imread(
            r'E:\py_project\wrsrobot\wrs_shu\0000_robot_grasp\dataset\bolts\dst_2\color_image_20251119-152415.jpg')
        if image is None:
            raise FileNotFoundError("图像读取失败，请检查路径")
        detector = ObjectDetector(obj_type='blot')  # 目标检测实例（对象）
        image_1 = image.copy()
        pos, vec, classes = detector.run_on_image(image_1, draw=False)
        if pos:
            print("本地图像坐标下的螺母位置:")
            for p in pos:
                print(f"({p[0]}, {p[1]}, {p[2]})")


    main()
