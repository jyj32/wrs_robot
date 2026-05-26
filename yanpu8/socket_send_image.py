import socket
import os
import time
import cv2
import numpy as np
import Mech_camera as mc

def save_images(camera, save_dir='./images/Mech'):
    rgb, depth = camera.capture_rgb_and_depth(save=False, show=False)
    if rgb is None or depth is None:
        return None, None

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    rgb_path = os.path.join(save_dir, f"rgb_{timestamp}.jpg")
    cv2.imwrite(rgb_path, rgb)

    depth_path = os.path.join(save_dir, f"depth_{timestamp}.npy")
    np.save(depth_path, depth)

    return rgb_path, depth_path

def handle_client(client_socket, camera):
    """处理单个客户端的命令循环"""
    try:
        while True:
            cmd = client_socket.recv(1024).decode().strip()
            if not cmd:
                break
            if cmd == "capture":
                rgb_path, depth_path = save_images(camera)
                if rgb_path and depth_path:
                    # 统一路径分隔符
                    rgb_path = rgb_path.replace('\\', '/')
                    depth_path = depth_path.replace('\\', '/')
                    # 关键：末尾加上换行符，让客户端知道消息结束
                    message = f"{rgb_path}\n{depth_path}\n"
                    client_socket.send(message.encode('utf-8'))
                    print(f"已发送路径：{rgb_path}, {depth_path}")
                else:
                    client_socket.send(b"FAIL\n")   # 也加换行符
            elif cmd == "exit":
                break
    except Exception as e:
        print(f"处理客户端异常: {e}")
    finally:
        client_socket.close()

def simple_server():
    camera = mc.CaptureImage(save_directory='./images/Mech')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 8888))
    server_socket.listen(5)   # 最大等待连接数
    print("服务器启动，监听 127.0.0.1:8888，等待客户端连接...")

    try:
        while True:   # 外层循环，持续接受新客户端
            client_socket, client_addr = server_socket.accept()
            print(f"客户端已连接：{client_addr}")
            handle_client(client_socket, camera)
            print(f"客户端 {client_addr} 已断开，等待下一个客户端...")
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    finally:
        server_socket.close()
        # 如果相机有关闭方法，可以调用 camera.disconnect()

if __name__ == "__main__":
    simple_server()