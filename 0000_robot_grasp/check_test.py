import cv2
import numpy as np

def detect_gaskets_interactive(image, roi):
    """
    带交互式参数调节的垫片检测函数
    通过Trackbar实时调整参数并查看效果
    """
    # 初始化参数
    params = {
        'threshold': 70,
        'dp': 12,  # 存储为整数(实际值=dp/10)
        'minDist': 5,
        'param1': 100,
        'param2': 30,
        'minRadius': 5,
        'maxRadius': 30
    }

    # 创建显示窗口
    cv2.namedWindow('Parameters')
    cv2.namedWindow('Detection Result')

    # 创建Trackbar回调函数
    def nothing(x):
        # 获取当前所有参数值
        params['threshold'] = cv2.getTrackbarPos('Threshold', 'Parameters')
        params['dp'] = max(1, cv2.getTrackbarPos('DP(x0.1)', 'Parameters'))
        params['minDist'] = cv2.getTrackbarPos('MinDist', 'Parameters')
        params['param1'] = max(1, cv2.getTrackbarPos('Param1', 'Parameters'))
        params['param2'] = max(1, cv2.getTrackbarPos('Param2', 'Parameters'))
        params['minRadius'] = cv2.getTrackbarPos('MinRadius', 'Parameters')
        params['maxRadius'] = cv2.getTrackbarPos('MaxRadius', 'Parameters')

        # 执行检测
        x1, x2 = roi['x']
        y1, y2 = roi['y']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[int(y1):int(y2), int(x1):int(x2)] = 255
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.GaussianBlur(gray, (5, 5), 2)
        _, thresh = cv2.threshold(blurred, params['threshold'], 255, cv2.THRESH_BINARY_INV)

        # 实际dp值为存储值/10
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT,
                                   dp=params['dp'] / 10,
                                   minDist=params['minDist'],
                                   param1=params['param1'],
                                   param2=params['param2'],
                                   minRadius=params['minRadius'],
                                   maxRadius=params['maxRadius'])

        # 显示结果
        output_img = image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0]:
                cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

        # 显示阈值图像和结果
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Detection Result', output_img)

    # 创建Trackbars
    cv2.createTrackbar('Threshold', 'Parameters', params['threshold'], 255, nothing)
    cv2.createTrackbar('DP(x0.1)', 'Parameters', params['dp'], 30, nothing)  # dp范围1.0-3.0
    cv2.createTrackbar('MinDist', 'Parameters', params['minDist'], 100, nothing)
    cv2.createTrackbar('Param1', 'Parameters', params['param1'], 200, nothing)
    cv2.createTrackbar('Param2', 'Parameters', params['param2'], 100, nothing)
    cv2.createTrackbar('MinRadius', 'Parameters', params['minRadius'], 100, nothing)
    cv2.createTrackbar('MaxRadius', 'Parameters', params['maxRadius'], 200, nothing)

    # 初始调用一次
    nothing(0)

    # 等待用户操作
    print("调整参数后按ESC退出...")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            break

    cv2.destroyAllWindows()
    return params


# 使用示例
if __name__ == "__main__":
    # 准备测试图像和ROI
    image = cv2.imread(r'F:\Study\point cloud\wrs-qiu\wrs-qiu\0000_grasp_concave\Data_Intel_Realsense_D405\color_image_20250609-142917.jpg')
    roi = {'x': (100, 400), 'y': (50, 300)}  # 根据实际情况调整

    # 运行交互式检测
    optimal_params = detect_gaskets_interactive(image, roi)

    print("\n最优参数配置：")
    print(f"阈值 threshold: {optimal_params['threshold']}")
    print(f"累加器分辨率 dp: {optimal_params['dp'] / 10:.1f}")
    print(f"最小间距 minDist: {optimal_params['minDist']}")
    print(f"高阈值 param1: {optimal_params['param1']}")
    print(f"累加器阈值 param2: {optimal_params['param2']}")
    print(f"最小半径 minRadius: {optimal_params['minRadius']}")
    print(f"最大半径 maxRadius: {optimal_params['maxRadius']}")