import time
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import matplotlib.pyplot as plt
import math
from concurrent.futures import ThreadPoolExecutor
import warnings

from yanpu_ur8.config import CONFIG_U1, CONFIG_U625

warnings.filterwarnings('ignore')   # 忽略警告

# U625yolo检测
class YOLOToMask:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"):
        """
        初始化YOLO检测器，用于将检测结果转换为mask

        Args:
            model_path: 训练好的模型路径（best.pt）
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        print(f"正在加载模型: {model_path}")
        # 使用更快的推理设置
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.save_dir = save_dir
        # 预热模型以加速首次推理
        if hasattr(self.model, 'predict'):
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, verbose=False)

        # 获取类别名称
        self.class_names = self.model.names
        print(f"模型加载成功！检测类别: {self.class_names}")

    @staticmethod
    def calculate_bbox_center(bbox):
        """计算边界框中心点（向量化操作）"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def calculate_distance_to_image_center(bbox_center, image_center):
        """计算边界框中心点到图像中心点的距离"""
        return math.hypot(bbox_center[0] - image_center[0],
                          bbox_center[1] - image_center[1])

    def load_image(self, image_path):
        """
        加载图像（优化版）
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
        # 延迟颜色转换，只在需要时进行
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_bbox_and_center_array(self, targets_info):
        """
        提取所有目标的bounding_box和bbox_center信息，返回NumPy数组

        Args:
            targets_info: 排序后的目标信息列表

        Returns:
            result_array: NumPy数组，每个元素包含[bounding_box, bbox_center]
        """
        result_list = []
        for target_info in targets_info:
            bounding_box = target_info.get('bbox', [])
            bbox_center = target_info.get('bbox_center', (0, 0))
            if bounding_box:
                result_list.append([bounding_box, bbox_center])

        return np.array(result_list, dtype=object)

    def _process_detection(self, idx, box, mask_data, w, h, image_center):
        """处理单个检测结果（用于并行处理）"""
        # 获取目标信息
        # cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # class_name = self.class_names[cls_id]

        # 获取边界框
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        bbox = [x1, y1, x2, y2]

        # 计算边界框中心点和距离
        bbox_center = self.calculate_bbox_center(bbox)
        distance = self.calculate_distance_to_image_center(bbox_center, image_center)

        # 快速处理mask
        mask_np = mask_data.cpu().numpy().astype(np.uint8)
        if mask_np.shape != (h, w):
            # 使用INTER_NEAREST以获得最快速度
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_np = (mask_np * 255).astype(np.uint8)

        return {
            'original_index': idx,
            # 'class': class_name,
            'confidence': conf,
            'bbox': bbox,   # 矩形边界框
            'bbox_center': bbox_center, # 矩形中心
            'distance': distance,   # 距离
            'mask': mask_np # 掩码图
        }

    def detect_and_save_masks(self, image_path=None, color_image=None,
                              last_point=CONFIG_U625['grasp']['mask_center'], grasp_success = True,
                            save=True, parallel=True):
        """
        检测图片，返回离参考点最近的掩码及其中心点

        Args:
            image_path: 输入图片路径
            color_image: 直接输入的图像数据（RGB格式）
            last_point: 参考点坐标 (x, y)，也是上次的目标坐标，默认为图像中心
            grasp_success: 上次是否抓取成功
            save: 是否保存文件（最近目标的掩码、合并掩码、可视化图）
            parallel: 是否使用并行处理加速

        Returns:
            mask_uint8: 二值掩码 (np.uint8, shape H×W)，若未检测到目标则为 None
            center: 边界框中心点 (cx, cy)，若未检测到目标则为 图片中心
        """
        # 读取图片
        if color_image is not None:
            image = color_image
        elif image_path is not None:
            if not os.path.exists(image_path):
                print(f"错误：图像路径不存在 - {image_path}")
                return None, None
            image = self.load_image(image_path)
            if image is None:
                print("错误：无法读取图像")
                return None, None
        else:
            print("错误：未提供图像输入")
            return None, None
        # 确定参考点
        if grasp_success:   # 上次抓取成功
            reference_point = CONFIG_U625['grasp']['mask_center']   # 选择箱子中心点
        else:   # 上次抓取失败
            reference_point = last_point    # 选择上次的点
        # YOLO 检测
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            max_det=300,
            augment=False,
            agnostic_nms=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            print("未检测到任何目标")
            return None, CONFIG_U625['grasp']['mask_center']
        if not hasattr(results, 'masks') or results.masks is None:
            print("错误: 模型没有输出mask，请确保使用的是分割模型")
            return None, CONFIG_U625['grasp']['mask_center']
        print(f"检测到 {len(results.boxes)} 个目标")

        h, w = image.shape[:2]
        # 处理检测结果
        boxes = results.boxes
        masks_data = results.masks.data
        if parallel and len(boxes) > 5: # 使用平行计算，或目标数量大于5
            # 对每个检测目标启动一个线程，调用self._process_detection方法, 计算该目标的掩码、边界框、中心点、距离等
            with ThreadPoolExecutor(max_workers=min(8, len(boxes))) as executor:
                futures = [
                    executor.submit(self._process_detection, i, box, masks_data[i], w, h, reference_point)
                    for i, box in enumerate(boxes)
                ]
                targets_info = [f.result() for f in futures]
        else:   # 一个一个计算
            targets_info = []
            for i, (box, mask_data) in enumerate(zip(boxes, masks_data)):
                targets_info.append(
                    self._process_detection(i, box, mask_data, w, h, reference_point)
                )   # 处理单张图片结果
        # 找最近目标
        closest_target = min(targets_info, key=lambda x: x['distance'])
        closest_mask = closest_target['mask']    # 掩码图
        # 保存文件（可选）
        if save:
            # 确保保存目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            # 时间
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # yolo分割结果图
            annotated_img = results.plot()  # BGR 格式，可直接保存
            vis_path = os.path.join(self.save_dir, f"yolo_seg_{timestamp}.jpg")
            cv2.imwrite(vis_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # 单个掩码图
            mask_filename = f"mask_{timestamp}.png"
            mask_path = os.path.join(self.save_dir, mask_filename)
            cv2.imwrite(mask_path, closest_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        # 只返回掩码和掩码中心点
        return closest_mask, closest_target['bbox_center']

# U1的yolo检测位置
class YOLODetector_objs:
    def __init__(self, yolo_path=None, save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech"):
        """
        初始化SAM模型
        参数:
            yolo_pth_path: 模型文件路径
        """
        if not os.path.exists(yolo_path):
            print(f"错误：模型文件不存在 {yolo_path}")
            return
        # 加载模型
        self.yolo_model = YOLO(yolo_path, verbose=False)
        print(f"YOLO分割模型加载完成")
        self.save_dir = save_dir

    # yolo实例分割，输出中间最近/面积最大的物体掩码和物体类型，用于远处拍摄
    def seg_objs_by_yolo_2(self, image, lastpoint=CONFIG_U1['grasp']['mask_center'], grasp_success=True, show=False,
                           save=True):
        """
        输入：
            image:rgb图
            lastpoint:上次的抓取点
            grasp_success:上次抓取是否成功
            show:是否展示
            save:是否保存
        输出：
            cls_id:物体正反面类型（0为反面，1为正面）
            mask_uint8:下次抓取的物体掩码
            mask_center:抓取中心点（图像上的坐标）
            cropped_image:裁剪后图片
        """
        # print("输入图像 shape:", image.shape)
        results = self.yolo_model(image, conf=0.7)
        result = results[0]
        if result.masks is None:    # 无掩码
            print("无掩码")
            return None,None,None,None
        if grasp_success:  # 抓取成功,找面积最大的掩码
            masks_data = result.masks.data
            if hasattr(masks_data, 'cpu'):  # 如果是 PyTorch 张量
                masks_data = masks_data.cpu().numpy()
            # 计算每个掩码的面积（像素数）
            mask_areas = masks_data.sum(axis=(1, 2))  # shape: (N,)
            # 找到面积最大的掩码索引
            largest_idx = mask_areas.argmax()
            # print(f"largest_idx:{largest_idx}")
            # 获取该掩码
            mask = masks_data[largest_idx]  # 二值数组，shape (H, W)
            print(f"最大掩码面积: {mask_areas[largest_idx]} 像素")
            if mask_areas[largest_idx] < 10000:  # 面积过小，检测出错
                print("检测掩码面积过小")
                return None, None, None, None
            h_orig, w_orig = image.shape[:2]
            # 如果模型推理时有填充（pad），需要裁剪掉填充部分
            if hasattr(result, 'pad') and result.pad is not None:
                l, t, r, b = result.pad
                mask = mask[t:t + h_orig, l:l + w_orig]  # 裁剪后应为 (h_orig, w_orig)
            else:
                # 否则直接缩放到原图尺寸
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            # 转换为 uint8 格式（0 或 255）
            mask_uint8 = (mask * 255).astype(np.uint8)
            # 求掩码中心坐标
            box = result.boxes[largest_idx]  # 对应的 Box 对象
            # 方法1：使用 xyxy 坐标计算中心
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 转为 Python 列表或数值
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # print(f"边界框中心（xyxy）: ({cx:.1f}, {cy:.1f})")
            mask_center = np.array([cx, cy])
            # 物体类别
            cls_id = int(box.cls[0])  # 类别索引
            cls_name = self.yolo_model.names[cls_id]  # 类别名称
            print(f"物体类别:{cls_id}即{cls_name}")
            # 裁剪掩码水平框图像
            x1_int = max(0, int(round(x1)))
            y1_int = max(0, int(round(y1)))
            x2_int = min(image.shape[1], int(round(x2)))
            y2_int = min(image.shape[0], int(round(y2)))
            # 创建全黑背景图（与原图尺寸相同）
            cropped_image = np.zeros_like(image)  # 全黑
            # 将框内区域复制到黑图上
            cropped_image[y1_int:y2_int, x1_int:x2_int] = image[y1_int:y2_int, x1_int:x2_int]
            # 如果需要显示结果
            if show:
                result_img = result.plot()  # 绘制所有检测结果
                # 在 result_img 上绘制目标掩码的水平矩形框
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 转换为整数
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 红色粗线
                # 将掩码转为三通道彩色以便拼接
                mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                # 统一宽度（以 result_img 的宽度为准）
                h1, w1 = result_img.shape[:2]
                h2, w2 = mask_color.shape[:2]
                if w1 != w2:
                    scale = w1 / w2
                    new_h = int(h2 * scale)
                    mask_color = cv2.resize(mask_color, (w1, new_h), interpolation=cv2.INTER_NEAREST)
                # 竖直拼接
                combined = cv2.vconcat([result_img, mask_color])
                # 缩放显示
                combined = cv2.resize(combined, (768, 960), interpolation=cv2.INTER_AREA)
                cv2.namedWindow('Largest Mask and Result', cv2.WINDOW_NORMAL)
                cv2.imshow('Largest Mask and Result', combined)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

        else:   # 抓取失败，找离上次抓取点最近的物体的掩码
            grasp_center = lastpoint
            min_distance = float('inf') # 无穷大
            nearest_idx = None  # 记录最近物体的索引
            nearest_corners = None
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):  # 遍历时记录索引 i
                    x1, y1, x2, y2 = box[:4]
                    area = abs(x2 - x1) * abs(y2 - y1)
                    if area < 10000: # 去除过小的掩码,一般情况下最底层大于15000万
                        print("检测到掩码过小")
                        continue
                    # print(f"远处拍摄掩码面积: {area}")
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center = np.array([center_x, center_y])
                    distance = np.linalg.norm(center - grasp_center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        nearest_idx = i  # 记录索引

            if nearest_idx is None: # 没有检测到物体
                return None,None,None,None
            else:   # 有检测到物体
                # 获取对应掩码，形状为 (H, W)，值为 float32 0~1
                mask = result.masks.data[nearest_idx].cpu().numpy()
                h_orig, w_orig = image.shape[:2]
                # 尝试用 pad 裁剪
                if hasattr(result, 'pad') and result.pad is not None:
                    l, t, r, b = result.pad
                    mask = mask[t:t + h_orig, l:l + w_orig]  # 裁剪后尺寸应为 (h_orig, w_orig)
                else:
                    # 否则 resize
                    mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                # 转换为 uint8 格式（0 或 255），便于后续处理或保存
                mask_uint8 = (mask * 255).astype(np.uint8)
                # 保存掩码
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                mask_filename = f"box_segmented_mask_{timestamp}.png"  # 掩码文件名
                mask_path = os.path.join(self.save_dir, mask_filename)  # 保存的掩码路径
                cv2.imwrite(mask_path, mask_uint8)  # 保存
                print(f"掩码已保存到: {mask_path}")
                # print("掩码 shape:", mask.shape)
                # 掩码中心点坐标
                x1, y1 = nearest_corners[0]  # 左上角
                x2, y2 = nearest_corners[2]  # 右下角（索引2对应右下）
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                mask_center = np.array([center_x, center_y])
                # 物体类别
                box = result.boxes[nearest_idx]
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                print(f"物体类别:{cls_id}即{cls_name}")
                # 裁剪图片
                x1_int = max(0, int(round(x1)))
                y1_int = max(0, int(round(y1)))
                x2_int = min(image.shape[1], int(round(x2)))  # 宽度是 shape[1]
                y2_int = min(image.shape[0], int(round(y2)))  # 高度是 shape[0]
                # 创建全黑背景图（与原图尺寸相同）
                cropped_image = np.zeros_like(image)  # 全黑
                # 将框内区域复制到黑图上
                cropped_image[y1_int:y2_int, x1_int:x2_int] = image[y1_int:y2_int, x1_int:x2_int]
                if show:
                    result_img = result.plot()  # 绘制所有检测结果
                    corners_int = nearest_corners.astype(int)
                    cv2.polylines(result_img, [corners_int], True, (0, 255, 0), 3)
                    # 将掩码转换为三通道彩色图像，便于拼接
                    mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                    # 统一宽度（以 result_img 的宽度为准，等比例缩放掩码）
                    h1, w1 = result_img.shape[:2]
                    h2, w2 = mask_color.shape[:2]
                    if w1 != w2:
                        # 计算缩放比例，保持宽高比
                        scale = w1 / w2
                        new_h = int(h2 * scale)
                        mask_color = cv2.resize(mask_color, (w1, new_h), interpolation=cv2.INTER_NEAREST)
                        print(f"掩码缩放至宽度 {w1}，新尺寸: {mask_color.shape[:2]}")
                    # 竖直拼接
                    combined = cv2.vconcat([result_img, mask_color])
                    # print("竖直拼接后 combined shape:", combined.shape)
                    # 缩放整体显示尺寸
                    combined = cv2.resize(combined, (768, 960), interpolation=cv2.INTER_AREA)
                    # print(f"缩放后combined.shape:{combined.shape}")
                    cv2.namedWindow('Result and Mask (Vertical)', cv2.WINDOW_NORMAL)
                    cv2.imshow('Result and Mask (Vertical)', combined)
                    cv2.waitKey(5000)
                    cv2.destroyAllWindows()
        if save:
            # 有掩码，保存带标注的图像
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.makedirs(self.save_dir, exist_ok=True)  # 用于创建目录
            save_path = os.path.join(self.save_dir, f"yolo_result_mech{timestamp}.jpg")
            result.save(filename=save_path)
            print(f"掩码检测结果已保存至: {save_path}")
            # 保存掩码
            mask_filename = f"mask_{timestamp}.png"
            mask_path = os.path.join(self.save_dir, mask_filename)
            cv2.imwrite(mask_path, mask_uint8)
            print(f"最大面积掩码已保存到: {mask_path}")
            # 保存裁剪图
            crop_filename = f"cropped_{timestamp}.jpg"
            crop_path = os.path.join(self.save_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_image)
            print(f"裁剪图像已保存到: {crop_path}")

        return cls_id, mask_uint8, mask_center, cropped_image

    def get_rotated_rect_from_mask(self, mask_uint8):
        """
        从二值掩码中提取最小外接旋转矩形信息。
        输入：
            mask_uint8: shape (H, W) 的 uint8 二值图像 (0 或 255)
        输出：
            字典，包含 'center', 'size', 'angle', 'corners'
        """
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # 取面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box_points = np.int32(cv2.boxPoints(rect))
        return {
            'center': rect[0],
            'size': rect[1],
            'angle': rect[2],
            'corners': box_points.tolist()
        }

# U1的yolo检测角度
class YOLODetector_angle:
    def __init__(self, yolo_path=None, save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech"):
        """
        初始化SAM模型
        参数:
            yolo_pth_path: 模型文件路径
        """
        if not os.path.exists(yolo_path):
            print(f"错误：模型文件不存在 {yolo_path}")
            return
        # 正确使用YOLO API加载模型
        self.yolo_model = YOLO(yolo_path, verbose=False)
        print(f"YOLO_obb模型加载完成")
        self.save_dir = save_dir

    def extract_obb_info(self, result):
        obb_info = []
        if result.obb is not None and len(result.obb) > 0:
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            cls_ids = result.obb.cls.cpu().numpy()
            for i in range(len(boxes)):
                corners = boxes[i].reshape(4, 2)
                info = {
                    'class_id': int(cls_ids[i]),
                    'class_name': result.names[int(cls_ids[i])],
                    'confidence': float(confs[i]),
                    'center': corners.mean(axis=0).tolist(),  # 中心点
                }
                obb_info.append(info)
        return obb_info

    def point_in_rotated_rect_by_vertices(self, point, vertices):
        """
        通过矩形的四个顶点判断点是否在旋转矩形内部（基于向量叉积法）。
        参数:
            point: tuple/list/np.ndarray, 点的坐标 (x, y)
            vertices: list/np.ndarray of shape (4,2), 矩形的四个顶点坐标（顺序任意，但通常为顺时针或逆时针）

        返回:
            bool: 点在内部返回 True，否则返回 False。
        """
        # 转换为 numpy 数组
        p = np.array(point)
        verts = np.array(vertices)
        # 确保顶点按顺序连接（例如逆时针），计算每个边与点的叉积符号一致性
        # 注意：旋转矩形是凸多边形，可以使用此法
        # 计算叉积符号
        signs = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            signs.append(np.sign(cross))  # sign 返回 -1, 0, 1
        # 如果所有符号相等（允许 0），则点在内部或边界上
        # 将 0 视为与任一符号相同，简化判断：排除既有正又有负的情况
        if (np.all(np.array(signs) >= 0) or np.all(np.array(signs) <= 0)):
            return True
        else:
            return False

    # 点绕中心点旋转angle角度后的坐标,计算出角2(范围-90，0，90，180)
    def point_rotation(self, point, center, angle):
        x, y = point
        x_center, y_center = center
        x1 = x - x_center
        y1 = y - y_center
        # print(f"(x1, y1):{(x1, y1)}")
        # 旋转后相对于中心点的坐标
        x2 = math.cos(angle) * x1 - math.sin(angle) * y1
        y2 = math.sin(angle) * x1 + math.cos(angle) * y1
        # print(f"(x2, y2):{(x2, y2)}")
        if x2 > 0 and y2 > 0:
            return math.pi
        elif x2 < 0 < y2:
            return math.pi/2
        elif x2 < 0 and y2 < 0:
            return 0
        else:
            return -math.pi/2

    def detect_angle_by_yolo(self,image, rotated_rect_info, show=True, save=True):  # YOLO检测零件角度
        """
        输入：
            image: 图片
            rotated_rect_info: 掩码的最小矩形的信息
            show: 是否展示图片
        输出：
            angle: 旋转角度，范围在[-270,180],从上往下看物体，顺时针为正，
        """
        results = self.yolo_model(image, conf=0.7)
        # result_img = results[0].plot()
        # cv2.imshow('Yolo Result', result_img)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()
        # results是一个列表，我们取第一个元素（因为只检测一张图片）
        if save:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            result = results[0]
            os.makedirs(self.save_dir, exist_ok=True)  # 用于创建目录
            save_path = os.path.join(self.save_dir, f"yolo_angle_result{timestamp}.jpg")
            result.save(filename=save_path)
            print(f"yolo角度检测结果已保存至: {save_path}")

        detections = self.extract_obb_info(results[0])
        center_np = np.array(rotated_rect_info['center'])   # 矩形中心坐标
        hole_list=[]    # 圆孔
        bump_list = []  # 凸起点
        for det in detections:
            if self.point_in_rotated_rect_by_vertices(det['center'], rotated_rect_info['corners']): # 在掩码矩形内
                if det['class_name'] == 'hole':
                    hole_list.append(det['center'])
                else:
                    bump_list.append(det['center'])
        # print(hole_list)
        # print(bump_list)
        if len(bump_list) >= 1 and len(hole_list) >= 4:
            print("凸起和圆孔都检测成功")
            # 凸起坐标
            if len(bump_list) >1:   # 凸起数量大于1，取离矩形中心最近的凸起
                print("凸起数量大于1")
                bump = min(bump_list, key=lambda p: np.linalg.norm(np.array(p) - center_np))    # 也可以取置信度最高的那1个
            else:   # 凸起数量为1
                bump = bump_list[0]
            if len(hole_list)>4:    # 圆孔数量大于4，取离矩形中心最近的凸起；也可以取置信度最高的那4个
                print("圆孔数量大于4")
                four_holes_list = sorted(hole_list, key=lambda p: np.linalg.norm(np.array(p) - center_np))[:4]
            else:   # 圆孔数量为4
                four_holes_list = hole_list
            # 计算最小外接旋转矩形
            rect = cv2.minAreaRect(np.array(four_holes_list, dtype=np.float32))
            # print(rect)
            # 获取矩形的四个角点
            box_points = np.intp(cv2.boxPoints(rect))
            # print(box_points)
            fi = -math.radians(rect[2]) # 角1(弧度制)，范围[-90,0]
            theta = self.point_rotation(bump, rect[0], fi)  # 角2,范围-180,-90,0,90
            print(f"角1:{math.degrees(fi)}°")
            print(f"角2:{math.degrees(theta)}°")
            angle_result = fi + theta
            print(f"最终旋转角:{math.degrees(angle_result)}°")
            if show:
                # 1. 获取 YOLO 检测结果图像
                result_img = results[0].plot()
                # 2. 绘制自定义的旋转矩形和角度
                image_with_rect = image.copy()
                cv2.drawContours(image_with_rect, [box_points], 0, (0, 255, 0), 2)
                for point in box_points:
                    cv2.circle(image_with_rect, tuple(point), 5, (0, 0, 255), -1)
                center = box_points.mean(axis=0).astype(int)
                cv2.circle(image_with_rect, tuple(center), 8, (255, 0, 0), -1)
                angle_text = f"Rotation: {math.degrees(angle_result):.2f}"
                text_position = center + np.array([10, 10])
                cv2.putText(image_with_rect, angle_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # 3. 水平拼接两幅图像
                combined = cv2.hconcat([result_img, image_with_rect])
                # 4. 将拼接后的图像缩小一半（宽高各缩放为原来的 0.5 倍）
                combined_resized = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                # 5. 显示缩小后的图像
                cv2.imshow('YOLO Detection & Custom Rectangle', combined_resized)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()

        elif len(hole_list) >= 4:   # 可以检测矩形方框，但凸起检测失败
            print("凸起检测失败，圆孔检测成功")
            if len(hole_list)>4:
                print("圆孔数量大于4")
                four_holes_list = sorted(hole_list, key=lambda p: np.linalg.norm(np.array(p) - center_np))[:4]
            else:  # 圆孔数量为4
                four_holes_list = hole_list
            # 计算最小外接旋转矩形
            rect = cv2.minAreaRect(np.array(four_holes_list, dtype=np.float32))
            # print(rect)
            # 获取矩形的四个角点
            box_points = np.intp(cv2.boxPoints(rect))
            # print(box_points)
            fi = -math.radians(rect[2])  # 角1(弧度制)，范围[-90,0]
            print(f"fi:{-rect[2]}")
            angle_result = fi
            width, height = rotated_rect_info['size']  # 获取掩码最小矩形的宽度和高度
            # print(f"width:{width}, height:{height}")
            # print(f"掩码的最小矩形的角度:{rotated_rect_info['angle']}")
            if rect[2]-30<rotated_rect_info['angle']<rect[2]+30: # 两个角度很接近，可以调整接近的阈值
                # print("两个角度相差不大")
                if width > height:  # 继续旋转90°
                    angle_result += np.pi / 2
            else:   # 两个角度相差很大，互余
                # print("两个角度相差很大")
                if width < height:  # 继续旋转90°
                    angle_result += np.pi / 2
            # print(f"angle_result:{angle_result}")
            if show:
                # 1. 获取 YOLO 检测结果图像（假设已经得到 result_img）
                result_img = results[0].plot()  # 这里确保 result_img 已定义
                # 2. 绘制自定义的旋转矩形和角度（您的现有代码）
                image_with_rect = image.copy()
                cv2.drawContours(image_with_rect, [box_points], 0, (0, 255, 0), 2)
                for point in box_points:
                    cv2.circle(image_with_rect, tuple(point), 5, (0, 0, 255), -1)
                center = box_points.mean(axis=0).astype(int)
                cv2.circle(image_with_rect, tuple(center), 8, (255, 0, 0), -1)
                angle_text = f"Rotation: {math.degrees(angle_result):.2f}"
                text_position = center + np.array([10, 10])
                cv2.putText(image_with_rect, angle_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # 3. 水平拼接两幅图像（要求高度相同）
                combined = cv2.hconcat([result_img, image_with_rect])
                combined_resized = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                # 4. 在单个窗口中显示
                cv2.imshow('YOLO Detection & Custom Rectangle', combined_resized)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

        elif len(bump_list)>=1: # 凸起检测成功但圆孔数量不对
            print("凸起检测成功但圆孔数量少了")
            if len(bump_list)>1:
                print("凸起数量大于1")
                bump = min(bump_list, key=lambda p: np.linalg.norm(np.array(p) - center_np))
            else:  # 凸起数量为1
                bump = bump_list[0]
            fi = -math.radians(rotated_rect_info['angle'])  # 掩码矩形旋转角取负号，角1(弧度制)，范围[-90,0]
            # 凸起绕掩码矩形中心旋转fi角，求得theta
            theta = self.point_rotation(bump, rotated_rect_info['center'], fi)  # 角2,范围：-90,0,90,180
            print(f"角1:{math.degrees(fi)}")
            print(f"角2:{math.degrees(theta)}")
            angle_result = fi + theta   # 范围,-180~180
            print(f"最终旋转角:{angle_result},即{math.degrees(angle_result)}°")
            if show:
                # 1. 获取 YOLO 检测结果图像（已包含所有标注）
                result_img = results[0].plot()
                image_with_rect = image.copy()
                corners_array = np.array(rotated_rect_info['corners'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.drawContours(image_with_rect, [corners_array], 0, (0, 255, 0), 2)
                # 绘制四个角点
                for point in rotated_rect_info['corners']:
                    cv2.circle(image_with_rect, tuple(point), 5, (0, 0, 255), -1)
                # 中心点转换为整数元组
                center = tuple(map(int, rotated_rect_info['center']))
                cv2.circle(image_with_rect, center, 8, (255, 0, 0), -1)
                # 文本位置使用整数坐标（避免元组+数组）
                angle_text = f"Rotation: {math.degrees(angle_result):.2f}"
                text_position = (center[0] + 10, center[1] + 10)
                cv2.putText(image_with_rect, angle_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # 拼接并显示
                combined = cv2.hconcat([result_img, image_with_rect])
                combined_resized = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow('YOLO Detection & Custom Rectangle', combined_resized)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

        else:   # 凸起和圆孔都检测失败
            print("凸起和圆孔都检测失败")
            fi = -math.radians(rotated_rect_info['angle'])  # 角1(弧度制)，范围[-90,0]
            print(f"fi:{math.degrees(fi)}度")
            angle_result = fi
            width, height = rotated_rect_info['size']  # 获取宽度和高度
            if width > height:  # 继续旋转90°
                angle_result +=  np.pi/2    # 范围-90~90
            if show:
                result_img = results[0].plot()
                image_with_rect = image.copy()
                corners_array = np.array(rotated_rect_info['corners'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.drawContours(image_with_rect, [corners_array], 0, (0, 255, 0), 2)
                # 绘制四个角点
                for point in rotated_rect_info['corners']:
                    cv2.circle(image_with_rect, tuple(point), 5, (0, 0, 255), -1)
                # 中心点转换为整数元组
                center = tuple(map(int, rotated_rect_info['center']))
                cv2.circle(image_with_rect, center, 8, (255, 0, 0), -1)
                # 文本位置使用整数坐标（避免元组+数组）
                angle_text = f"Rotation: {math.degrees(angle_result):.2f}"
                text_position = (center[0] + 10, center[1] + 10)
                cv2.putText(image_with_rect, angle_text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # 拼接并显示
                combined = cv2.hconcat([result_img, image_with_rect])
                combined_resized = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # 缩放
                cv2.imshow('YOLO Detection & Custom Rectangle', combined_resized)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
        return angle_result

class YOLO_detect_exist_U1():  # 检测区域内物体是否存在
    def __init__(self, yolo_path=None, save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"):
        """
                初始化SAM模型
                参数:
                    yolo_pth_path: 模型文件路径
                """
        if not os.path.exists(yolo_path):
            print(f"错误：YOLO检测U1是否存在的模型文件不存在 {yolo_path}")
            return
        # 正确使用YOLO API加载模型
        self.yolo_model = YOLO(yolo_path, verbose=False)
        print(f"YOLO_obb模型加载完成")
        self.save_dir = save_dir
    def detect_exist_U1(self,image,x_range=(130, 270),y_range=(300, 450),save=False,show=False):
        """
        检测图片中的矩形区域内是否有物体U1
        输入：
            image:图片
            x_range:x范围
            y_range:y范围
            save:是否保存检测结果
            show:是否展示
        输出：
            exists:存在/不存在
        """
        # 0.处理图片
        range_image = np.zeros_like(image)  # 全黑
        # 将箱子框内区域复制到黑图上，中心坐标（宽1000，高475）
        range_image[y_range[0]:y_range[1], x_range[0]:x_range[1]] = image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        # 1. 推理图像
        results = self.yolo_model(range_image, verbose=False, conf=0.7)
        if len(results) == 0 or results[0].boxes is None or results[0].boxes.xyxy.numel() == 0:
            return False
        # 2.获取所有检测框 (xyxy格式: x1, y1, x2, y2)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
        # 3. 检查是否存在大小合适的检测框
        exists = False
        for box in boxes:
            x1, y1, x2, y2 = box
            area = abs((x2 - x1) * (y2 - y1))
            print(f"放置区域物体框面积：{area}")
            if 4000 < area < 7000:   # 一般情况下5708
                exists = True
                break
        # 4. 保存标注后的结果图像
        if save and exists:
            # 获取带标注的图像（YOLO的plot方法返回BGR图像）
            annotated_img = results[0].plot()
            # 生成唯一文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.save_dir, f"detect_U1_in_place{timestamp}.jpg")
            cv2.imwrite(save_path, annotated_img)
            print(f"检测结果已保存至: {save_path}")
            if show:
                cv2.imshow("Detection Result", annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return exists

class YOLO_detect_exist_U625():  # 检测区域内物体是否存在
    def __init__(self, yolo_path=None, save_dir=r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"):
        """
                初始化SAM模型
                参数:
                    yolo_pth_path: 模型文件路径
                """
        if not os.path.exists(yolo_path):
            print(f"错误：YOLO检测U1是否存在的模型文件不存在 {yolo_path}")
            return
        # 正确使用YOLO API加载模型
        self.yolo_model = YOLO(yolo_path, verbose=False)
        print(f"YOLO_obb模型加载完成")
        self.save_dir = save_dir
    def detect_exist_U625(self,image,x_range=(450,640),y_range=(0,150),save=True,show=False):
        """
        检测图片中的矩形区域内是否有物体U1
        输入：
            image:图片
            x_range:x范围
            y_range:y范围
            save:是否保存检测结果
            show:是否展示
        输出：
            exists:存在/不存在
        """
        # 0.处理图片
        range_image = np.zeros_like(image)  # 全黑
        # 将箱子框内区域复制到黑图上，中心坐标（宽1000，高475）
        range_image[y_range[0]:y_range[1], x_range[0]:x_range[1]] = image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        # 1. 推理图像
        results = self.yolo_model(range_image, verbose=False, conf=0.7)
        if len(results) == 0 or results[0].boxes is None or results[0].boxes.xyxy.numel() == 0:
            return False
        # 2.获取所有检测框 (xyxy格式: x1, y1, x2, y2)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
        # 3. 检查是否存在大小合适的检测框
        exists = False
        for box in boxes:
            x1, y1, x2, y2 = box
            area = abs((x2 - x1) * (y2 - y1))
            print(f"放置区域物体框面积：{area}")
            if 4000 < area < 5500:   # 4816,
                exists = True
                break
        # 4. 保存标注后的结果图像
        if save and exists:
            # 获取带标注的图像（YOLO的plot方法返回BGR图像）
            annotated_img = results[0].plot()
            # 生成唯一文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.save_dir, f"detect_U625_in_place{timestamp}.jpg")
            cv2.imwrite(save_path, annotated_img)
            print(f"检测结果已保存至: {save_path}")
            if show:
                cv2.imshow("Detection Result", annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            # 获取带标注的图像（YOLO的plot方法返回BGR图像）
            annotated_img = results[0].plot()
            # 生成唯一文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.save_dir, f"detect_U625_in_place{timestamp}.jpg")
            cv2.imwrite(save_path, annotated_img)
            print(f"检测结果已保存至: {save_path}")

        return exists


# ========== 使用示例 ==========
if __name__ == "__main__":
    # # 设置模型路径
    # model_path = r"C:\Users\cc\Documents\Github\ultralytics-main\runs\train\exp\weights\best.pt"
    #
    # # 初始化转换器
    # converter = YOLOToMask(
    #     model_path=model_path,
    #     conf_threshold=0.25,
    #     iou_threshold=0.45
    # )
    #
    # # 测试单张图片
    # image_path = r"C:\Users\cc\Documents\Github\cc-wrs\chenchen\yanpu_ur\image4\color_image_20260226-134345.jpg"
    #
    # # 运行检测（可以设置parallel=False来关闭并行以观察速度差异）
    # sorted_result_array, mask_files = converter.detect_and_save_masks_sorted(
    #     image_path=image_path,
    #     output_dir=r"C:\Users\cc\Desktop\extracted_masks_sorted",
    #     save_visualization=False,
    #     parallel=True  # 启用并行处理
    # )
    #
    # if sorted_result_array is not None:
    #     print(f"\n结果数组形状: {sorted_result_array.shape}")
    #     if len(sorted_result_array) > 0:
    #         print(f"第一个目标中心点: {sorted_result_array[0][1]}")
    #     print(f"生成的mask文件数: {len(mask_files)}")

    # image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech\color_image_20260525-133615.jpg"
    # image = cv2.imread(image_path)
    # # 创建全黑背景图（与原图尺寸相同）
    # box_image = np.zeros_like(image)  # 全黑
    # # 将箱子框内区域复制到黑图上
    # box_image[0:950, 300:1700] = image[0:950, 300:1700]   # 高，宽
    # cv2.imshow('image', box_image)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    # # 保存
    # crop_filename = f"box_image.jpg"
    # save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\Mech"
    # crop_path = os.path.join(save_dir, crop_filename)
    # cv2.imwrite(crop_path, box_image)
    # print(f"裁剪图像已保存到: {crop_path}")

    #
    # image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\color_image_20260514-122044.jpg"
    # image = cv2.imread(image_path)
    # # 创建全黑背景图（与原图尺寸相同）
    # box_image = np.zeros_like(image)  # 全黑
    # # 将箱子框内区域复制到黑图上
    # box_image[100:390, 110:540] =image[100:390, 110:540]
    # cv2.imshow('image', box_image)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    # # 保存
    # crop_filename = f"box_image.jpg"
    # save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"
    # crop_path = os.path.join(save_dir, crop_filename)
    # cv2.imwrite(crop_path, box_image)
    # print(f"裁剪图像已保存到: {crop_path}")

    # # 检测U625的分割
    # image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\color_image_20260507-191622.jpg"
    # image = cv2.imread(image_path)
    # model_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\yolo_pths\bestutest.pt"
    #
    # # 初始化转换器
    # converter = YOLOToMask(
    #     model_path=model_path,
    #     conf_threshold=0.25,
    #     iou_threshold=0.45,
    #     save_dir= r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"
    # )
    # mask, mask_center = converter.detect_and_save_masks(
    #     image_path=image_path,
    #     color_image=image,
    #     last_point=np.array([350,100]), # 宽，高
    #     grasp_success=False, # 是否抓取成功
    #     save=True
    # )
    import D435camera as D435

    save_dir = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"
    D435_2 = D435.D435Detector(camera_type='d435', save_directory=save_dir, camera_serial='317222074435')
    model_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\yolo_pths\best6.pt"

    U625_detect = YOLO_detect_exist_U625(
        yolo_path=model_path,
        save_dir=r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"
    )
    start_time = time.time()
    image = D435_2.capture_rgb(0.1, False, True)
    # 检测区域内有无物体
        # image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\color_image2_20260526-193740.jpg"
    # 初始化转换器

    # image = cv2.imread(image_path)
    exist = U625_detect.detect_exist_U625(image,(450,640),(0,150),False,False)
    # print(exist)
    print(f"{time.time()-start_time}")
    #
    # model_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\yolo_pths\best5.pt"
    # # image_path = r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435\color_image2_20260526-193740.jpg"
    # # 初始化转换器
    # U1_detect = YOLO_detect_exist_U1(
    #     yolo_path=model_path,
    #     save_dir=r"E:\py_project\wrsrobot\wrs_shu\yanpu_ur8\images\D435"
    # )
    # # image = cv2.imread(image_path)
    # exist = U1_detect.detect_exist_U1(image,(130,270),(300,450),True,True)
    # print(exist)

