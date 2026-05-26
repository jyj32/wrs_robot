#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 梅卡曼德相机外参标定
'''
@Project ：wrs-fujikoshi
@File ：demo1_Finding_transformation_matrix_optimization_hjy.py
@IDE ：PyCharm
@Author ：hujia
@Date ：2025/7/8 16:28
'''

import tkinter as tk
import numpy as np
import open3d as o3d
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.ur7e.ur7e_withouttable as ur7
from panda3d.core import AmbientLight
from direct.task import Task
from panda3d.core import CollisionTraverser, CollisionNode, CollisionSphere, CollisionRay
from panda3d.core import CollisionHandlerQueue, BitMask32
import math


class UR7EDemo:
    def __init__(self, root):
        # 保存 Tkinter 根窗口
        self.root = root

        # 初始化 Panda3D 世界
        self.base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0])
        print("Panda3D 环境初始化完成")

        # 添加光源和显示坐标系
        self.add_light()
        gm.gen_frame(pos=[0, 0, 0], rotmat=np.eye(3), length=0.2, thickness=0.005).attach_to(self.base)

        # 初始化 ur7 机器人并添加到 Panda3D 场景中
        self.setup_robot()

        # 用于存储当前显示的点云对象
        self.pointcloud_obj = None
        self.axis_obj = None  # 用于显示旋转轴

        # 键盘交互相关变量
        self.key_states = {}  # 存储按键状态

        # 灵敏度相关变量
        self.base_rotation_sensitivity = 2.0  # 基础旋转灵敏度
        self.base_translation_sensitivity = 0.01  # 基础平移灵敏度
        self.sensitivity_multiplier = 1.0  # 灵敏度倍数
        self.min_sensitivity = 0.1  # 最小灵敏度倍数
        self.max_sensitivity = 5.0  # 最大灵敏度倍数
        self.sensitivity_step = 0.1  # 灵敏度调节步长

        # 点云变换参数
        self.current_rotation = [0, 0, 0]  # alpha, beta, gamma
        self.current_translation = [0, 0, 0]  # tx, ty, tz
        self.pcd_center = np.array([0, 0, 0])  # 点云中心点（用于显示旋转轴）

        # 设置键盘事件
        self.setup_keyboard_events()

        # 将 Tkinter 的更新任务添加到 Panda3D 的任务管理器中
        self.base.taskMgr.add(self.tk_update_task, "tk_update")

    @property
    def rotation_sensitivity(self):
        """当前旋转灵敏度"""
        return self.base_rotation_sensitivity * self.sensitivity_multiplier

    @property
    def translation_sensitivity(self):
        """当前平移灵敏度"""
        return self.base_translation_sensitivity * self.sensitivity_multiplier

    def adjust_sensitivity(self, increase=True):
        """调节灵敏度"""
        if increase:
            self.sensitivity_multiplier = min(self.max_sensitivity,
                                              self.sensitivity_multiplier + self.sensitivity_step)
        else:
            self.sensitivity_multiplier = max(self.min_sensitivity,
                                              self.sensitivity_multiplier - self.sensitivity_step)

        # 更新GUI显示
        if hasattr(self, 'control_window'):
            self.control_window.update_sensitivity_display()

        print(f"灵敏度调节至: {self.sensitivity_multiplier:.1f}x")

    def tk_update_task(self, task):
        # 更新 Tkinter 窗口事件
        self.root.update()
        return Task.cont

    def setup_robot(self):
        # 初始化 gofa5 机器人
        self.robot = ur7.UR7E(enable_cc=True)
        self.robot.hnd.jaw_to(0)
        self.robot.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(self.base)
        print("ur7e 机器人初始化完成")

    def add_light(self):
        # 添加环境光源
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color((1, 1, 1, 1))
        ambient_node = self.base.render.attach_new_node(ambient_light)
        self.base.render.set_light(ambient_node)
        print("环境光源添加完成")

    def setup_keyboard_events(self):
        """设置键盘事件监听"""
        # 键盘按下事件
        self.base.accept("w", self.set_key_state, ["w", True])
        self.base.accept("w-up", self.set_key_state, ["w", False])
        self.base.accept("s", self.set_key_state, ["s", True])
        self.base.accept("s-up", self.set_key_state, ["s", False])
        self.base.accept("a", self.set_key_state, ["a", True])
        self.base.accept("a-up", self.set_key_state, ["a", False])
        self.base.accept("d", self.set_key_state, ["d", True])
        self.base.accept("d-up", self.set_key_state, ["d", False])
        self.base.accept("q", self.set_key_state, ["q", True])
        self.base.accept("q-up", self.set_key_state, ["q", False])
        self.base.accept("e", self.set_key_state, ["e", True])
        self.base.accept("e-up", self.set_key_state, ["e", False])

        # 方向键
        self.base.accept("arrow_up", self.set_key_state, ["arrow_up", True])
        self.base.accept("arrow_up-up", self.set_key_state, ["arrow_up", False])
        self.base.accept("arrow_down", self.set_key_state, ["arrow_down", True])
        self.base.accept("arrow_down-up", self.set_key_state, ["arrow_down", False])
        self.base.accept("arrow_left", self.set_key_state, ["arrow_left", True])
        self.base.accept("arrow_left-up", self.set_key_state, ["arrow_left", False])
        self.base.accept("arrow_right", self.set_key_state, ["arrow_right", True])
        self.base.accept("arrow_right-up", self.set_key_state, ["arrow_right", False])

        # Z/C 控制Z轴平移
        self.base.accept("z", self.set_key_state, ["z", True])
        self.base.accept("z-up", self.set_key_state, ["z", False])
        self.base.accept("c", self.set_key_state, ["c", True])
        self.base.accept("c-up", self.set_key_state, ["c", False])

        # 重置键
        self.base.accept("r", self.reset_transform)

        # 输出最终数值的键
        self.base.accept("p", self.print_final_values)

        # 灵敏度调节键
        self.base.accept("x", self.decrease_sensitivity)  # x键降低灵敏度
        self.base.accept("space", self.increase_sensitivity)  # 空格键提高灵敏度

        # 添加键盘处理任务
        self.base.taskMgr.add(self.keyboard_task, "keyboard_task")

        print("控制说明:")
        print("WASD: 控制点云绕X/Y轴旋转（相对世界原点）")
        print("QE: 控制点云绕Z轴旋转（相对世界原点）")
        print("方向键: 控制点云XY平移")
        print("Z/C: 控制点云Z轴平移（上/下）")
        print("R: 重置所有变换")
        print("P: 输出当前旋转和平移数值")
        print("空格: 提高操作灵敏度")
        print("X: 降低操作灵敏度")

    def increase_sensitivity(self):
        """提高灵敏度"""
        self.adjust_sensitivity(increase=True)

    def decrease_sensitivity(self):
        """降低灵敏度"""
        self.adjust_sensitivity(increase=False)

    def set_key_state(self, key, state):
        """设置按键状态"""
        self.key_states[key] = state

    def keyboard_task(self, task):
        """键盘处理任务"""
        changed = False

        # 获取当前灵敏度
        current_rotation_sensitivity = self.rotation_sensitivity
        current_translation_sensitivity = self.translation_sensitivity

        # 处理旋转
        if self.key_states.get("w", False):  # 绕X轴正向旋转
            self.current_rotation[0] += current_rotation_sensitivity
            self.show_rotation_axis('X')
            changed = True
        if self.key_states.get("s", False):  # 绕X轴反向旋转
            self.current_rotation[0] -= current_rotation_sensitivity
            self.show_rotation_axis('X')
            changed = True
        if self.key_states.get("a", False):  # 绕Y轴正向旋转
            self.current_rotation[1] += current_rotation_sensitivity
            self.show_rotation_axis('Y')
            changed = True
        if self.key_states.get("d", False):  # 绕Y轴反向旋转
            self.current_rotation[1] -= current_rotation_sensitivity
            self.show_rotation_axis('Y')
            changed = True
        if self.key_states.get("q", False):  # 绕Z轴正向旋转
            self.current_rotation[2] += current_rotation_sensitivity
            self.show_rotation_axis('Z')
            changed = True
        if self.key_states.get("e", False):  # 绕Z轴反向旋转
            self.current_rotation[2] -= current_rotation_sensitivity
            self.show_rotation_axis('Z')
            changed = True

        # 处理平移
        if self.key_states.get("arrow_up", False):  # Y轴正向平移
            self.current_translation[1] += current_translation_sensitivity
            changed = True
        if self.key_states.get("arrow_down", False):  # Y轴反向平移
            self.current_translation[1] -= current_translation_sensitivity
            changed = True
        if self.key_states.get("arrow_left", False):  # X轴反向平移
            self.current_translation[0] -= current_translation_sensitivity
            changed = True
        if self.key_states.get("arrow_right", False):  # X轴正向平移
            self.current_translation[0] += current_translation_sensitivity
            changed = True
        if self.key_states.get("z", False):  # Z轴正向平移（向上）
            self.current_translation[2] += current_translation_sensitivity
            changed = True
        if self.key_states.get("c", False):  # Z轴反向平移（向下）
            self.current_translation[2] -= current_translation_sensitivity
            changed = True

        # 限制角度和平移范围
        for i in range(3):
            self.current_rotation[i] = max(-180, min(180, self.current_rotation[i]))
            self.current_translation[i] = max(-2.0, min(2.0, self.current_translation[i]))

        # 如果没有按键被按下，隐藏旋转轴
        rotation_keys_pressed = any(self.key_states.get(key, False) for key in ["w", "s", "a", "d", "q", "e"])
        if not rotation_keys_pressed:
            self.hide_rotation_axis()

        # 如果有变化，更新点云和GUI
        if changed:
            self.update_point_cloud_from_keyboard()
            # 更新控制窗口的滑块（如果存在）
            if hasattr(self, 'control_window'):
                self.control_window.update_sliders_from_values()

        return Task.cont

    def show_rotation_axis(self, axis):
        """显示旋转轴（现在显示世界坐标系的轴）"""
        # 移除之前的轴显示
        self.hide_rotation_axis()

        # 创建轴线（绕世界原点的轴）
        origin = np.array([0, 0, 0])  # 世界原点

        if axis == 'X':
            # X轴 - 红色
            start_pos = origin + np.array([-0.5, 0, 0])
            end_pos = origin + np.array([0.5, 0, 0])
            color = [1, 0, 0, 1]
        elif axis == 'Y':
            # Y轴 - 绿色
            start_pos = origin + np.array([0, -0.5, 0])
            end_pos = origin + np.array([0, 0.5, 0])
            color = [0, 1, 0, 1]
        elif axis == 'Z':
            # Z轴 - 蓝色
            start_pos = origin + np.array([0, 0, -0.5])
            end_pos = origin + np.array([0, 0, 0.5])
            color = [0, 0, 1, 1]

        # 创建轴线几何体
        self.axis_obj = gm.gen_stick(spos=start_pos, epos=end_pos, thickness=0.01, rgba=color)
        self.axis_obj.attach_to(self.base)

    def hide_rotation_axis(self):
        """隐藏旋转轴"""
        if self.axis_obj:
            self.axis_obj.detach()
            self.axis_obj = None

    def reset_transform(self):
        """重置所有变换"""
        self.current_rotation = [0, 0, 0]
        self.current_translation = [0, 0, 0]
        self.update_point_cloud_from_keyboard()
        if hasattr(self, 'control_window'):
            self.control_window.update_sliders_from_values()
        print("变换已重置")

    def print_current_pose(self):
        """实时输出当前位姿"""
        print("=" * 30)
        print("当前点云位姿:")
        print(
            f"旋转角度 (度): X={self.current_rotation[0]:.2f}, Y={self.current_rotation[1]:.2f}, Z={self.current_rotation[2]:.2f}")
        print(
            f"平移距离: X={self.current_translation[0]:.4f}, Y={self.current_translation[1]:.4f}, Z={self.current_translation[2]:.4f}")
        print("=" * 30)

    def print_final_values(self):
        """输出最终的旋转和平移数值（按P键触发）"""
        print("\n" + "=" * 50)
        print("最终变换参数（相对世界原点）:")
        print("-" * 20)
        print("旋转参数 (欧拉角，单位：度):")
        print(f"  Alpha (绕X轴): {self.current_rotation[0]:.3f}°")
        print(f"  Beta  (绕Y轴): {self.current_rotation[1]:.3f}°")
        print(f"  Gamma (绕Z轴): {self.current_rotation[2]:.3f}°")
        print("-" * 20)
        print("平移参数 (单位：米):")
        print(f"  X轴平移: {self.current_translation[0]:.6f}")
        print(f"  Y轴平移: {self.current_translation[1]:.6f}")
        print(f"  Z轴平移: {self.current_translation[2]:.6f}")
        print("-" * 20)
        print("旋转参数 (弧度):")
        print(f"  Alpha: {np.pi * self.current_rotation[0] / 180:.6f}")
        print(f"  Beta:  {np.pi * self.current_rotation[1] / 180:.6f}")
        print(f"  Gamma: {np.pi * self.current_rotation[2] / 180:.6f}")
        print("-" * 20)
        print("变换矩阵:")
        rotation_matrix = rm.rotmat_from_euler(
            np.pi * self.current_rotation[0] / 180,
            np.pi * self.current_rotation[1] / 180,
            np.pi * self.current_rotation[2] / 180
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = self.current_translation
        print("4x4 变换矩阵:")
        for row in transform_matrix:
            print(f"  [{row[0]:8.6f}, {row[1]:8.6f}, {row[2]:8.6f}, {row[3]:8.6f}]")
        print("=" * 50 + "\n")

    def add_point_cloud(self, pcd_path):
        # 使用 Open3D 加载点云文件
        pcd = o3d.io.read_point_cloud(pcd_path)
        self.pcd_np = np.asarray(pcd.points)

        # 过滤无效数据
        self.pcd_np = self.pcd_np[~np.isnan(self.pcd_np).any(axis=1)]
        print(f"点云加载成功,包含 {len(self.pcd_np)} 个有效点")
        # self.pcd_np = self.pcd_np/1000

        # 计算点云中心点（仅用于显示目的）
        self.pcd_center = np.mean(self.pcd_np, axis=0)
        print(f"点云中心点(米): {self.pcd_center}")
        print("注意：旋转将相对于世界原点 (0,0,0) 进行")

        # 检查点云是否包含颜色数据
        if pcd.has_colors():
            colors_np = np.asarray(pcd.colors)  # 形状 (N, 3)，值范围 [0,1]
            # 添加 alpha 通道（不透明度设为 1）
            colors_np = np.concatenate((colors_np, np.ones((len(colors_np), 1))), axis=1)  # (N, 4)
            # 关键修改：将 NumPy 数组转换为 Python 列表
            self.pcd_colors = colors_np.tolist()
            print("点云包含颜色数据")
        else:
            self.pcd_colors = [[1, 0, 0, 1]] * len(self.pcd_np)
            # self.pcd_colors = np.array([[1, 0, 0, 1]] * len(self.pcd_np))
            print("点云不包含颜色数据,使用默认红色")

        # 显示原始点云
        self.update_point_cloud_from_keyboard()

    def update_point_cloud_from_keyboard(self):
        """从键盘交互更新点云"""
        self.update_point_cloud(
            self.current_rotation[0],
            self.current_rotation[1],
            self.current_rotation[2],
            self.current_translation[0],
            self.current_translation[1],
            self.current_translation[2]
        )

    def update_point_cloud(self, alpha, beta, gamma, tx, ty, tz):
        """
        更新点云位置和旋转（相对于世界原点）

        Args:
            alpha: 绕X轴旋转角度（度）
            beta: 绕Y轴旋转角度（度）
            gamma: 绕Z轴旋转角度（度）
            tx, ty, tz: 平移量
        """
        # 旋转和平移应用（相对于世界原点）
        rotation_matrix = rm.rotmat_from_euler(np.pi * alpha / 180, np.pi * beta / 180, np.pi * gamma / 180)

        # 直接绕世界原点旋转，然后应用平移
        # 不再进行中心点偏移，直接对原始点云坐标进行旋转
        pcd_rotated = self.pcd_np @ rotation_matrix.T
        pcd_np_transformed = pcd_rotated + np.array([tx, ty, tz])

        # 如果存在旧点云对象,移除它
        if self.pointcloud_obj:
            self.pointcloud_obj.detach()

        # 显示旋转和平移后的点云
        self.pointcloud_obj = gm.gen_pointcloud(pcd_np_transformed, rgbas=self.pcd_colors, pntsize=3)
        self.pointcloud_obj.attach_to(self.base)

    def run_panda3d(self):
        # 保持 Panda3D 窗口打开
        self.base.run()

class ControlWindow:
    def __init__(self, root, ur7_demo):
        self.ur7_demo = ur7_demo
        self.ur7_demo.control_window = self  # 建立反向引用
        self.root = root

        # 添加灵敏度显示框架
        sensitivity_frame = tk.Frame(self.root)
        sensitivity_frame.pack(pady=5)

        tk.Label(sensitivity_frame, text="操作灵敏度:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.sensitivity_label = tk.Label(sensitivity_frame, text=f"{self.ur7_demo.sensitivity_multiplier:.1f}x",
                                          font=("Arial", 12, "bold"), fg="blue")
        self.sensitivity_label.pack(side=tk.LEFT, padx=10)

        # 添加灵敏度调节按钮
        tk.Button(sensitivity_frame, text="降低灵敏度 (X)",
                  command=self.ur7_demo.decrease_sensitivity).pack(side=tk.LEFT, padx=5)
        tk.Button(sensitivity_frame, text="提高灵敏度 (空格)",
                  command=self.ur7_demo.increase_sensitivity).pack(side=tk.LEFT, padx=5)

        # 添加使用说明
        instruction_frame = tk.Frame(self.root)
        instruction_frame.pack(pady=5)

        instructions = [
            "控制说明（旋转相对于世界原点）:",
            "键盘操作:",
            "  WASD: 控制点云绕X/Y轴旋转（相对世界原点）",
            "  QE: 控制点云绕Z轴旋转（相对世界原点）",
            "  方向键: 控制点云XY平移",
            "  Z/C: 控制点云Z轴平移（上/下）",
            "  R: 重置所有变换",
            "  P: 输出当前变换数值",
            "灵敏度控制:",
            "  空格: 提高操作灵敏度",
            "  X: 降低操作灵敏度"
        ]

        for instruction in instructions:
            color = "blue" if "控制说明" in instruction else (
                "green" if any(x in instruction for x in ["键盘操作", "灵敏度控制"]) else "black")
            label = tk.Label(instruction_frame, text=instruction,
                             font=("Arial", 9), fg=color)
            label.pack(anchor="w")

        # 创建旋转角度和平移的滑块+输入框组合
        self.create_slider_and_entry("X 轴旋转角度 (alpha) ° (相对世界原点)", -180, 180, 0, "alpha")
        self.create_slider_and_entry("Y 轴旋转角度 (beta) ° (相对世界原点)", -180, 180, 0, "beta")
        self.create_slider_and_entry("Z 轴旋转角度 (gamma) ° (相对世界原点)", -180, 180, 0, "gamma")
        self.create_slider_and_entry("X 轴平移 (tx)", -2.0, 2.0, 0, "tx")
        self.create_slider_and_entry("Y 轴平移 (ty)", -2.0, 2.0, 0, "ty")
        self.create_slider_and_entry("Z 轴平移 (tz)", -2.0, 2.0, 0, "tz")

        # 添加按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="重置变换", command=self.reset_transform).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="输出数值", command=self.ur7_demo.print_final_values).pack(side=tk.LEFT, padx=5)

    def update_sensitivity_display(self):
        """更新灵敏度显示"""
        self.sensitivity_label.config(text=f"{self.ur7_demo.sensitivity_multiplier:.1f}x")

    def create_slider_and_entry(self, label, from_, to, initial, var_name):
        frame = tk.Frame(self.root)
        frame.pack(pady=5)

        tk.Label(frame, text=label).pack(side=tk.LEFT)

        # 创建滑块,使用 command 参数来监听滑块改变
        slider = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                          resolution=0.01 if "t" in var_name else 1,
                          command=lambda value, vn=var_name: self.slider_to_entry(vn, value))
        slider.set(initial)
        slider.pack(side=tk.LEFT)
        setattr(self, f"{var_name}_slider", slider)

        # 创建输入框
        entry = tk.Entry(frame, width=8)
        entry.insert(0, str(initial))
        entry.pack(side=tk.LEFT)
        setattr(self, f"{var_name}_entry", entry)

        # 绑定输入框的 Enter 键事件
        entry.bind("<Return>", lambda event, vn=var_name: self.entry_to_slider(vn))

    def slider_to_entry(self, var_name, value):
        # 滑块的值更新到输入框
        entry = getattr(self, f"{var_name}_entry")
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        self.update_point_cloud()

    def entry_to_slider(self, var_name):
        # 输入框的值更新到滑块
        entry = getattr(self, f"{var_name}_entry")
        slider = getattr(self, f"{var_name}_slider")
        try:
            value = float(entry.get())
            slider.set(value)
            self.update_point_cloud()
        except ValueError:
            pass  # 忽略无效输入

    def update_point_cloud(self):
        # 更新点云的旋转和位置
        alpha = float(self.alpha_entry.get())
        beta = float(self.beta_entry.get())
        gamma = float(self.gamma_entry.get())
        tx = float(self.tx_entry.get())
        ty = float(self.ty_entry.get())
        tz = float(self.tz_entry.get())

        # 同步更新内部状态
        self.ur7_demo.current_rotation = [alpha, beta, gamma]
        self.ur7_demo.current_translation = [tx, ty, tz]

        self.ur7_demo.update_point_cloud(alpha, beta, gamma, tx, ty, tz)

    def update_sliders_from_values(self):
        """从内部状态更新滑块值"""
        # 更新滑块和输入框
        self.alpha_slider.set(self.ur7_demo.current_rotation[0])
        self.beta_slider.set(self.ur7_demo.current_rotation[1])
        self.gamma_slider.set(self.ur7_demo.current_rotation[2])
        self.tx_slider.set(self.ur7_demo.current_translation[0])
        self.ty_slider.set(self.ur7_demo.current_translation[1])
        self.tz_slider.set(self.ur7_demo.current_translation[2])

        # 更新输入框
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, f"{self.ur7_demo.current_rotation[0]:.1f}")
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, f"{self.ur7_demo.current_rotation[1]:.1f}")
        self.gamma_entry.delete(0, tk.END)
        self.gamma_entry.insert(0, f"{self.ur7_demo.current_rotation[2]:.1f}")
        self.tx_entry.delete(0, tk.END)
        self.tx_entry.insert(0, f"{self.ur7_demo.current_translation[0]:.3f}")
        self.ty_entry.delete(0, tk.END)
        self.ty_entry.insert(0, f"{self.ur7_demo.current_translation[1]:.3f}")
        self.tz_entry.delete(0, tk.END)
        self.tz_entry.insert(0, f"{self.ur7_demo.current_translation[2]:.3f}")

    def reset_transform(self):
        """重置所有变换"""
        self.ur7_demo.current_rotation = [0, 0, 0]
        self.ur7_demo.current_translation = [0, 0, 0]
        self.update_sliders_from_values()
        self.ur7_demo.update_point_cloud_from_keyboard()


if __name__ == '__main__':
    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("点云旋转和平移调整 - 相对世界原点旋转")

    # 初始化 Panda3D 显示窗口
    ur7_demo = UR7EDemo(root)

    # 加载点云路径
    pcd_path = r'E:\py_project\wrsrobot\wrs_shu\yanpu_ur5\images\Mech\pointcloud.ply'

    ur7_demo.add_point_cloud(pcd_path)

    # 创建控制窗口
    control_window = ControlWindow(root, ur7_demo)

    # 主线程运行 Panda3D 窗口
    ur7_demo.run_panda3d()
