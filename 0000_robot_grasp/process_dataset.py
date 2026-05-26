import os
import cv2
from check_object import ObjectDetector
from realsense_camera import RealSenseCamera
# 拍图片数据集

if __name__ == '__main__':
    # src_dataset_dir = 'dataset/gasket/src'
    src_dataset_dir = 'dataset/bolts/src_2'
    # src_dataset_dir = 'dataset/nuts/src'
    # dst_dataset_dir = 'dataset/gasket/dst'
    dst_dataset_dir = 'dataset/bolts/dst_2'
    # dst_dataset_dir = 'dataset/nuts/dst'
    thresh_dataset_dir = 'dataset/bolts/thresh_2'
    # thresh_dataset_dir = 'dataset/nuts/thresh'
    os.makedirs(src_dataset_dir, exist_ok=True)
    os.makedirs(dst_dataset_dir, exist_ok=True)
    os.makedirs(thresh_dataset_dir, exist_ok=True)
    camera = RealSenseCamera(camera_type='d435', save_directory=src_dataset_dir)  # 启动RealSense D405相机
    detector = ObjectDetector(obj_type='gasket')
    camera.start()
    camera.interactive_capture()

    for root, dirs, files in os.walk(src_dataset_dir):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dataset_dir, file)
                    thresh_path = os.path.join(thresh_dataset_dir, file)
                    if not os.path.exists(dst_path):
                        src_image = cv2.imread(src_path)
                        copy_src = src_image.copy()
                        # 检测ArUco标记
                        aruco_px, img_marked = detector.detect_aruco_pixels(src_image, draw=False)

                        if not aruco_px or not all(k in aruco_px for k in [0, 1, 2, 3]):
                            print("缺少构建坐标系所需的 ArUco 标记 (0,1,2,3)")
                            continue
                        warped_image, M = detector.warp_to_marker_frame(copy_src, aruco_px, output_size=(460, 340)) # 对应实际尺寸为23cm*17cm
                        cv2.imwrite(dst_path, warped_image)
                        print(f"已保存仿射变换处理后的图片:{dst_path}")
                    else:
                        warped_image = cv2.imread(dst_path)
                    # if os.path.exists(thresh_path):
                    #     continue
                    # else:
                    #     detect_area = {'x': [30, 430], 'y': [0, 340]}
                    #     thresh_image, _ = detector.detect_gaskets_pixels(warped_image, detect_area, draw=False, threshold=100, param2=23)
                    #     cv2.imwrite(thresh_path, thresh_image)
                    #     print(f"已保存二值化处理后的图片:{thresh_path}")
                except Exception as e:
                    print(f"处理图片{os.path.join(root, file)}时发生错误:{e}")

