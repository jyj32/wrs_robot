import pyrealsense2 as rs
from typing import Literal
import numpy as np
import os
import cv2
import time


class RealSenseCamera:
    def __init__(self, camera_type: Literal['d405', 'd435'] = 'd435', save_directory='Data_Intel_Realsense_d435',
                 width=640, height=480, fps=30, warmup_frames=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.started = False
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        if camera_type.lower() == 'd405':
            self.camera_matrix = np.array([[434.44, 0., 322.235],
                                           [0., 433.249, 236.842],
                                           [0., 0., 1.]])
            self.dist_coeffs = np.array([[-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724]])
            self.depth_scale = 9.999999747378752e-05
        elif camera_type.lower() == 'd435':
            self.camera_matrix = np.array([[606.627, 0., 324.281],
                                           [0., 606.657, 241.149],
                                           [0., 0., 1.]])
            self.dist_coeffs = np.array([0., 0., 0., 0., 0.])
            self.depth_scale = 0.0010000000474974513
        else:
            raise ValueError('Either d405 or d435')

    def start(self):
        if self.started:
            return
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline.start(self.config)

        # 预热，跳过初始帧
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames()
        self.started = True

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

    def _undistorted_image(self, image):
        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), alpha=1)
        map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_K, (w, h),
                                                 cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return undistorted_img

    def capture(self):
        if not self.started:
            raise RuntimeError("请先调用 start() 启动相机")
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("未获取到有效帧")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = self._undistorted_image(color_image)
        return color_image

    def interactive_capture(self):
        if not self.started:
            raise RuntimeError("请先调用 start() 启动相机")
        print("相机已启动，按 Enter 拍照，按 q 退出...")
        last_saved_color_path = None
        try:
            i=0
            while True:
                color_image = self.capture()
                cv2.imshow('Color Image', color_image)
                key = cv2.waitKey(1)
                if key == 13:  # Enter
                    i += 1
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.png')  # 使用无损压缩png
                    cv2.imwrite(color_path, color_image)
                    print(f"已保存第{i}张图像：{color_path}")
                    last_saved_color_path = color_path
                elif key & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()

        if last_saved_color_path:
            print(f"使用最后保存的图像：{last_saved_color_path}")
            image = cv2.imread(last_saved_color_path)
            return image, last_saved_color_path
        else:
            print("没有保存图像")
            return None, None


if __name__ == '__main__':
    camera = RealSenseCamera('d435')
    camera.start()
    camera.interactive_capture()