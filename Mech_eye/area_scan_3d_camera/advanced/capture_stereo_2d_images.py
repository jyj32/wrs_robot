# With this sample, you can obtain and save the stereo 2D images.
# 通过该示例，您可以获取并保存立体二维图像。

import os.path
import cv2
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect

# Define the save directory
save_directory = r"G:\mecheye_python_samples\area_scan_3d_camera\advanced\dataset"


#  捕获立体二维图像
class CaptureStereo2DImages(object):
    def __init__(self):
        self.camera = Camera()

    def capture_stereo_2d_images(self):
        stereo_left = Frame2D()
        stereo_right = Frame2D()
        error = self.camera.capture_stereo_2d(stereo_left, stereo_right)

        if not error.is_ok():
            show_error(error)
            return

        # 根据捕获图像的颜色类型（单色或彩色），分别使用 get_gray_scale_image()
        # 或 get_color_image() 方法获取灰度图像或彩色图像。
        if stereo_left.color_type() == ColorTypeOf2DCamera_Monochrome:
            image_left = stereo_left.get_gray_scale_image()
            image_right = stereo_right.get_gray_scale_image()

        elif stereo_right.color_type() == ColorTypeOf2DCamera_Color:
            image_left = stereo_left.get_color_image()
            image_right = stereo_right.get_color_image()

        # image_file_left = "stereo2D_left.png"
        # image_file_right = "stereo2D_right.png"

        image_file_left = os.path.join(save_directory, f"stereo2D_left.png")
        image_file_right = os.path.join(save_directory, f"stereo2D_right.png")

        cv2.imshow(image_file_left, image_left.data())
        cv2.imshow(image_file_right, image_right.data())
        cv2.waitKey(0)

        cv2.imwrite(image_file_left, image_left.data())
        cv2.imwrite(image_file_right, image_right.data())

        print("Capture and save the stereo 2D images: {} and {}".format(
            image_file_left, image_file_right))

    def main(self):
        if find_and_connect(self.camera):
            self.capture_stereo_2d_images()
            self.camera.disconnect()
            print("Disconnected from the Stereo_Camera successfully.")


if __name__ == '__main__':
    a = CaptureStereo2DImages()
    a.main()
