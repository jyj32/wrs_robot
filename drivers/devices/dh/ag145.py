import drivers.devices.dh.dh_modbus_gripper as dh
import time
# 大寰（DH）夹爪的驱动程序

class Ag145driver():
    def __init__(self, port = 'com4', baudrate = 115200, force = 100, speed = 100):
        self.port = port
        self.baudrate = baudrate
        initstate = 0
        force = force
        speed = speed
        self.m_gripper = dh.dh_modbus_gripper()
        self.m_gripper.open(self.port, self.baudrate)
        self.init_gripper()
        while (initstate != 1):
            # 获取初始状态
            initstate = self.m_gripper.GetInitState()
            time.sleep(0.2)
        self.m_gripper.SetTargetPosition(500)
        self.set_speed(speed)
        self.set_force(force)

    def init_gripper(self):
        self.m_gripper.Initialization()
        print('Send grip init')

    def set_force(self, force=100):
        self.m_gripper.SetTargetForce(force)

    def set_speed(self, speed=100):
        self.m_gripper.SetTargetSpeed(speed)

    def conv2encoder(self, jawwidth):   # 0-100转换为0-1000
        return int(jawwidth*10)

    def check_connection(self):
        """
        检查夹爪连接状态
        返回: True 正常, False 断开
        """
        try:
            # 尝试读取状态，如果返回None说明通信失败
            state = self.m_gripper.GetInitState()
            return state is not None
        except Exception as e:
            print(f"连接检查异常: {e}")
            return False

    # def jaw_to(self, jawwidth):
    #     self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
    #     g_state = 0
    #     start_time = time.time()
    #     while g_state == 0:   # 0为手爪在运动
    #         # 超时检查
    #         if time.time() - start_time > 2:
    #             print(f"警告：夹爪移动超时！当前状态: {g_state}")
    #             break  # 强制退出
    #         g_state = self.m_gripper.GetGripState()
    #         time.sleep(0.02)

    def jaw_to(self, jawwidth):
        print("手爪动作")
        wide = int(self.conv2encoder(jawwidth))
        wide = max(0, wide)
        wide = min(1000, wide)
        self.m_gripper.SetTargetPosition(wide)
        start_time = time.time()
        while True:
            g_state = self.m_gripper.GetGripState()
            if g_state != 0:
                break
            if time.time() - start_time > 2:
                print("夹爪移动超时")
                break
            time.sleep(0.02)

    def open_g(self):
        self.jaw_to(0.06)

    def close_g(self):
        self.jaw_to(0)

    def get_current_width(self):    # 输出宽度0~100
        encoder = self.m_gripper.GetCurrentPosition()  # 返回 0~1000
        width = encoder / 10  # 范围0~100
        return width

if __name__ == '__main__':
    gripper = Ag145driver()
    gripper.open_g()



