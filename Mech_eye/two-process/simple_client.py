import socket
import numpy as np
import robot_con.ur.ur7_dh50_rtde as ur7con
import time


def simple_client():
    # 创建 socket 对象
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 连接服务器
    host = '127.0.0.1'
    port = 8888
    client_socket.connect((host, port))
    print(f"已连接到服务器 {host}:{port}")
    
    # 等待一段时间，确保其他程序可能已经断开连接
    time.sleep(2)
    
    try:
        rbt_r = ur7con.UR5Ag95X_RTDE(robot_ip='192.168.125.30',
                                     gp_port='COM5')
        print("机器人连接成功")
    except RuntimeError as e:
        print(f"机器人连接失败：{e}")
        print("请检查:")
        print("1. 是否有其他程序正在使用机器人？")
        print("2. 之前的程序是否正确断开连接？")
        client_socket.close()
        return

    try:
        while True:
            # 发送消息
            xxx = rbt_r.get_jnt_values()
            print(f"获取关节值：{xxx}")
            xxxx  = np.array([ 0.60689878, -1.37338159, 1.56827099, -1.75802054, -1.57196075,  0  ])
            rbt_r.move_jnts(xxxx, vel=0.1,acc=0.1)
            time.sleep(2)
            xxxx  = np.array([ 0.60689878, -1.37338159, 1.56827099, -1.75802054, -1.57196075,  0.61657   ])
            rbt_r.move_jnts(xxxx, vel=0.1,acc=0.1)
            # 接收响应
            response = client_socket.recv(1024).decode('utf-8')
            if not response:
                print("服务器已断开连接")
                break
            print(f"服务器响应：{response}")
            
    except KeyboardInterrupt:
        print("\n程序中断")
    except Exception as e:
        print(f"\n发生异常：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rbt_r.disconnect()
        client_socket.close()
        print("连接已关闭")


if __name__ == "__main__":
    simple_client()