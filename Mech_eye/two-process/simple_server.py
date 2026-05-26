import socket
import Mech_eye.Mech_camera as mc
import time


def simple_server():
    # 创建 socket 对象
    camera = mc.CaptureImage(save_directory='./images')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定地址和端口
    host = '127.0.0.1'
    port = 8888
    server_socket.bind((host, port))

    # 监听连接
    server_socket.listen(1)
    print(f"服务器启动，监听 {host}:{port}")

    # 接受客户端连接
    client_socket, client_address = server_socket.accept()
    print(f"客户端已连接：{client_address}")

    # # 初始化相机预览（只调用一次）
    # print("启动相机预览...")
    # camera.start_preview_optimized(preview_mode='rgb',
    #                                window_name='RGB Preview',
    #                                display_size=(640, 480),
    #                                show_fps=True,
    #                                print_time=True)

    try:
        while True:
            # 接收数据（非阻塞方式）
            try:
                # data = client_socket.recv(1024).decode('utf-8')
                # if not data:
                #     print("客户端已断开连接")
                #     break
                #
                # print(f"客户端：{data}")
                # 拍照
                image = camera.capture_point_cloud()
                # 发送响应
                response = image
                # response = f"服务器收到：{data}"
                client_socket.send(response.encode('utf-8'))
                
            except socket.error as e:
                print(f"Socket 错误：{e}")
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        print("\n程序中断")
    finally:
        camera.disconnect()
        print("相机已断开")
        client_socket.close()
        server_socket.close()
        print("服务器已关闭")


if __name__ == "__main__":
    simple_server()