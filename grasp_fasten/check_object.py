import os
import time
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from typing import Literal
from ultralytics import YOLO
from realsense_camera import RealSenseCamera
from config import CONFIG

import basis.robot_math as rm


class ObjectDetector:
    def __init__(self, aruco_dict=None, aruco_params=None, obj_type: Literal['gasket', 'blot', 'nut'] = 'gasket', size: Literal['M5', 'M3','M2']='M3'):
        marker_0_pos = np.array([0, 0])
        marker_1_pos = marker_0_pos + np.array([CONFIG['aruco']['marker_x_dist'], 0])
        marker_2_pos = marker_0_pos + np.array([0, CONFIG['aruco']['marker_y_dist']])
        marker_3_pos = marker_0_pos + np.array([CONFIG['aruco']['marker_x_dist'], CONFIG['aruco']['marker_y_dist']])

        self.marker_real_coords = {0: marker_0_pos, 1: marker_1_pos, 2: marker_2_pos, 3: marker_3_pos}
        self.detect_obj_name = obj_type
        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco_params or aruco.DetectorParameters()
        yolo_pth_path = f'./yolo_pths/{self.detect_obj_name}/best_{size}.pt'    # yolo模型路径
        self.size = size    # 物体尺寸
        self.yolo_model = YOLO(yolo_pth_path)

    def detect_aruco_pixels(self, image, draw=True):
        """
            获取二维码位置
        """
        if CONFIG['aruco']['use_default']:  # 是否使用自定义的二维码位置
            marker_px = CONFIG['default_aruco_px']
            return marker_px, image
        # 不使用自定义二维码位置
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测二维码及其位置
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
            # 绘制图像
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
        # 展示图像
        if draw:
            cv2.imshow("ArUco Detection", image)
            cv2.waitKey(0)  # 按任意键关闭窗口
        aruco.drawDetectedMarkers(image, corners, ids)
        print(f"marker_px:{marker_px}")
        return marker_px, image

    @staticmethod
    def warp_to_marker_frame(image, aruco_px_dict, output_size=(460, 340), save = False):
        """
        使用四个 ArUco 标记进行透视变换，将 marker 围成的区域变为矩形。
        默认顺序为：
            ID 0 - 左下（相对于机器人）
            ID 1 - 右下（相对于机器人）
            ID 2 - 左上（相对于机器人）
            ID 3 - 右上（相对于机器人）
        id和位置一定要固定对应，需要detector.detect_aruco_pixels(src_image, draw=True)函数绘制确认
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
        if save:
            # 创建保存目录
            save_dir = r'E:\py_project\wrsrobot\wrs_shu\grasp_fasten\images'
            os.makedirs(save_dir, exist_ok=True)
            # 生成时间戳文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(save_dir, f"warped_{timestamp}.png")
            cv2.imwrite(save_path, warped)
            print(f"仿射变换图片已保存至: {save_path}")

        return warped, M

    # yolo检测
    def detect_objs_by_yolo(self, image):

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

    def pixel_to_real(self, px, output_size=(460, 340)):
        """在仿射图中计算像素点相对 ID0 的实际物理坐标,高度为0"""

        dx_real = np.linalg.norm(self.marker_real_coords[1] - self.marker_real_coords[0])  # 通常是 0.23m
        dy_real = np.linalg.norm(self.marker_real_coords[2] - self.marker_real_coords[0])  # 通常是 0.17m
        w, h = output_size

        # 将像素转换为相对于 ID0 的物理坐标
        scale_x = dx_real / w
        scale_y = dy_real / h
        image_coord_x = px[0] * scale_x
        image_coord_y = px[1] * scale_y
        image_coord_z = 0
        image_coord = np.array([image_coord_x, image_coord_y, image_coord_z])
        convert_pos = CONFIG['aruco']['marker_0_real_pos']
        return image_coord + convert_pos

    @staticmethod
    def plot_result(original_image, result_image, aruco_px_dict):
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

    def get_blot_head_pos(self, image, pts, pixel_to_real_func=True, show=True):
        """
        可视化螺钉头部判断逻辑。
        参数：
            image: 输入图像（BGR 或灰度）
            pts: 矩形四个点，形状 (4, 2)
            pixel_to_real_func: 可选转换函数（像素->真实坐标）
            show: 是否显示 matplotlib 图像
        返回：
            bolt_head_pos: 螺栓头位置（真实坐标或像素坐标）
            bolt_tail_pos: 螺栓尾位置（真实坐标或像素坐标）
        """
        image_vis = image.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        pts = np.array(pts, dtype=np.float32).reshape(4, 2)

        def edge_length(pt1, pt2):  # 长度
            return np.linalg.norm(pt1 - pt2)

        def pts_center(pt1, pt2):   # 像素中心
            return np.mean(np.array([pt1, pt2]), axis=0)

        def get_mean_gray(rect_pts):    # 计算平均灰度
            mask = np.zeros_like(gray, dtype=np.uint8)
            rect_pts = rect_pts.astype(np.int32)
            cv2.fillPoly(mask, [rect_pts], 255) # type: ignore
            mean_val = cv2.mean(gray, mask=mask)[0]
            return mean_val
        # 计算矩形边长
        edge1 = edge_length(pts[0], pts[1])
        edge2 = edge_length(pts[1], pts[2])

        if edge1 > edge2:  # 横向
            # 分区，左区&右区
            lft_region = np.array([pts[0], pts_center(pts[0], pts[1]), pts_center(pts[2], pts[3]), pts[3]])
            rgt_region = np.array([pts_center(pts[0], pts[1]), pts[1], pts[2], pts_center(pts[2], pts[3])])
            # 计算区域的平均灰度
            lft_mean = get_mean_gray(lft_region)
            rgt_mean = get_mean_gray(rgt_region)
            # 选择灰度大的那头为selected，灰度小的那头为target
            selected = pts_center(pts[0], pts[3]) if lft_mean > rgt_mean else pts_center(pts[1], pts[2])
            target = pts_center(pts[1], pts[2]) if lft_mean > rgt_mean else pts_center(pts[0], pts[3])
        else:  # 竖向
            # 分区，上区（左区）&下区（右区）
            lft_region = np.array([pts[1], pts_center(pts[1], pts[2]), pts_center(pts[0], pts[3]), pts[0]])
            rgt_region = np.array([pts_center(pts[1], pts[2]), pts[2], pts[3], pts_center(pts[0], pts[3])])
            # 计算区域平均灰度
            lft_mean = get_mean_gray(lft_region)
            rgt_mean = get_mean_gray(rgt_region)
            # 选择灰度大的那头为selected，灰度小的那头为target
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
        # 螺栓头坐标
        blot_head_pos = self.pixel_to_real(selected) if pixel_to_real_func else selected
        # 螺栓尾坐标
        bolt_tail_pos = self.pixel_to_real(target) if pixel_to_real_func else target
        return blot_head_pos, bolt_tail_pos

    def get_vec_along_shot_edge(self, corners):
        '''
            给定一个矩形，计算出它的长边的垂直方向，即短边方向向量
            输入:
                corners:矩形的四个角点
            输出:
                short_unit:三维的方向向量，z方向为0
        '''
        d01 = np.linalg.norm(corners[0] - corners[1])
        d12 = np.linalg.norm(corners[1] - corners[2])
        # 比较出短边，并转换为世界坐标系的三维向量
        if d01 > d12:
            short_vec = self.pixel_to_real(corners[2]) - self.pixel_to_real(corners[1])
        else:
            short_vec = self.pixel_to_real(corners[1]) - self.pixel_to_real(corners[0])
        # 单位化
        short_unit = short_vec / np.linalg.norm(short_vec)
        return short_unit

    def get_center_and_vec_in_real(self, image, centers, corners, classes):
        """
            将 YOLO 检测到的物体的像素信息转换为真实世界坐标系下的抓取点位置和夹爪接近方向向量
            输入:
                image:图片
                centers:物体中心坐标
                corners:物体旋转矩形四个角点的像素坐标，形状 (4, 2)
                classes:类别名称字符串
            输出:
                center_list:抓取点的三维真实坐标
                vec_list:夹爪指向的方向向量
        """
        center_list = []    # 抓取点的三维真实坐标
        vec_list = []   # 夹爪指向的方向向量
        if self.detect_obj_name == 'gasket':    # 平垫圈
            for i, class_name in enumerate(classes):
                # 计算三维坐标
                center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['gasket'][self.size]['grip_z']]))
                # 计算方向
                if class_name == 'stack': # 堆叠
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))   # 沿短边方向
                else:   # 单个
                    vec_list.append(np.array([0, 1, 0]))    # 默认为y轴方向
        elif self.detect_obj_name == 'blot':    # 螺栓
            for i, class_name in enumerate(classes):
                if class_name == 'single':  # 单个
                    blot_head, blot_tail = self.get_blot_head_pos(image, corners[i], show=False) + np.array([0, 0, CONFIG['blot'][self.size]['grip_z']])
                    grasp_center = blot_head
                    grasp_vec = rm.unit_vector(blot_tail - blot_head)   # 从螺栓头指向螺栓尾的单位向量
                elif class_name == 'stack': # 堆叠
                    grasp_center = self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['blot'][self.size]['split_stack_z']])
                    grasp_vec = self.get_vec_along_shot_edge(corners[i])    # 沿短边方向的方向向量
                elif class_name == 'single_standing':   # 单个直立
                    grasp_center = self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['blot'][self.size]['standing_z']])
                    grasp_vec = np.array([0, 1, 0]) # 默认为y轴方向
                else:
                    raise Exception('Unknown class: {}'.format(class_name))
                center_list.append(grasp_center)
                vec_list.append(grasp_vec)
        elif self.detect_obj_name == 'nut': # 螺母
            for i, class_name in enumerate(classes):
                if class_name == 'stack':   # 堆叠
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['nut'][self.size]['grip_z']]))
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))
                elif class_name == 'single_standing':   # 单个直立
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['nut'][self.size]['split_stack_z']]))
                    vec_list.append(self.get_vec_along_shot_edge(corners[i]))
                else:   # 单个
                    center_list.append(self.pixel_to_real(centers[i]) + np.array([0, 0, CONFIG['nut'][self.size]['grip_z']]))
                    vec_list.append(np.array([1, 0, 0]))

        return center_list, vec_list

    def run_on_image(self, image, draw):
        """
        识别并分类图像中的目标物体，返回真实世界坐标系下的中心点坐标和方向向量
        输入:
            image:图片
            draw:是否展示
        输出:
            single_obj:中心位置和方向向量
            stack_obj:中心位置和方向向量
            standing_obj:中心位置和方向向量
        """
        # 检测二维码区域
        aruco_px, img_marked = self.detect_aruco_pixels(image, draw=draw)
        if not aruco_px or not all(k in aruco_px for k in [0, 1, 2, 3]):
            print("缺少构建坐标系所需的 ArUco 标记 (0,1,2,3)")
            return []
        # 仿射变换
        warped_image, M = self.warp_to_marker_frame(image, aruco_px, output_size=(460, 340),save = True)
        # yolo检测
        result_img, obj_corners, obj_centers, obj_classes = self.detect_objs_by_yolo(warped_image)
        self.plot_result(image, result_img, aruco_px)   # 展示结果
        obj_center_list, obj_vec_list = self.get_center_and_vec_in_real(warped_image, obj_centers, obj_corners, obj_classes)
        single_obj, standing_obj, stack_obj = [], [], []
        for i, obj_class in enumerate(obj_classes):
            if obj_class.lower() == 'single':
                single_obj.append([obj_center_list[i], obj_vec_list[i]])
            elif obj_class.lower() == 'single_standing':
                standing_obj.append([obj_center_list[i], obj_vec_list[i]])
            elif obj_class.lower() == 'stack':
                stack_obj.append([obj_center_list[i], obj_vec_list[i]])
        return single_obj, standing_obj, stack_obj


if __name__ == "__main__":

    # image = cv2.imread(
    #     r'F:\Study\point cloud\wrs-qiu\wrs-qiu\0000_grasp_concave\Data_Intel_Realsense_d405\color_image_20250627-145742.jpg')
    # if image is None:
    #     raise FileNotFoundError("图像读取失败，请检查路径")
    cam = RealSenseCamera()
    cam.start()
    image = cam.capture()
    detector = ObjectDetector(obj_type='blot', size = 'M2')
    image_1 = image.copy()
    # yolo检测
    single_obj, stack_obj, standing_obj = detector.run_on_image(image_1, draw=False)
    print(single_obj)
    print(stack_obj)
    print(standing_obj)


# {1: array([     521.75,      398.75], dtype=float32), 3: array([        519,        84.5], dtype=float32), 0: array([       92.5,       405.5], dtype=float32), 2: array([       86.5,       89.25], dtype=float32)}
