# socket_get_image.py
import socket
import cv2
import numpy as np
import time
# 接收图片

class CameraClient:
    def __init__(self, server_ip='127.0.0.1', server_port=8888):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = None
        self.connect()

    def connect(self):
        """建立 socket 连接"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.server_port))
        print(f"已连接到服务器 {self.server_ip}:{self.server_port}")

    def _recv_all(self, suffix=b'\n', buffer_size=4096):
        """接收数据直到遇到 suffix（换行符）"""
        data = b''
        while True:
            chunk = self.sock.recv(buffer_size)
            if not chunk:
                break
            data += chunk
            if data.endswith(suffix):
                break
        return data

    def capture(self):
        """
        发送一次 capture 命令，返回 (rgb_image, depth_array)
        如果失败返回 (None, None)
        """
        try:
            self.sock.send(b"capture")
            raw_data = self._recv_all()
            if not raw_data:
                return None, None
            msg = raw_data.decode().strip()
            if msg == "FAIL":
                print("服务器返回 FAIL")
                return None, None
            lines = msg.split('\n')
            if len(lines) != 2:
                raise ValueError(f"路径数据格式错误，行数: {len(lines)}")
            rgb_path, depth_path = lines[0], lines[1]
            # 路径中的反斜杠统一为正斜杠（Windows兼容）
            rgb_path = rgb_path.replace('\\', '/')
            depth_path = depth_path.replace('\\', '/')
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise IOError(f"无法读取 RGB 图片: {rgb_path}")
            depth = np.load(depth_path)
            return rgb, depth
        except Exception as e:
            print(f"拍照失败: {e}")
            return None, None

    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 保留旧的 simple_client 函数以便兼容（可选）
def simple_client():
    """旧接口：一次性连接、拍照、关闭，返回 (rgb, depth)"""
    client = CameraClient()
    rgb, depth = client.capture()
    client.close()
    return rgb, depth

if __name__ == "__main__":
    cc = CameraClient()
    for i in range(3):
        rgb, depth = cc.capture()
        time.sleep(5)