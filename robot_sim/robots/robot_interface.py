import copy
import numpy as np
import robot_sim._kinematics.collision_checker as cc


class RobotInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robot_interface'):
        # TODO self.jlcs = {}
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # collision detection
        self.cc = None
        # component map for quick access
        self.manipulator_dict = {}
        self.ft_sensor_dict = {}
        self.hnd_dict = {}

    def change_name(self, name):
        self.name = name

    def get_hnd_on_manipulator(self, manipulator_name):
        raise NotImplementedError

    def get_jnt_ranges(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_ranges()

    def get_jnt_values(self, component_name):
        return self.manipulator_dict[component_name].get_jnt_values()

    def get_gl_tcp(self, manipulator_name):
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        return self.manipulator_dict[component_name].is_jnt_values_in_ranges(jnt_values)

    def fix_to(self, pos, rotmat):
        return NotImplementedError

    def fk(self, component_name, jnt_values):
        return NotImplementedError

    def jaw_to(self, hnd_name, jaw_width):
        self.hnd_dict[hnd_name].jaw_to(jaw_width=jaw_width)

    def get_jawwidth(self, hand_name):
        return self.hnd_dict[hand_name].get_jawwidth()

    # 逆运动学求解
    def ik(self,
           component_name: str = "arm", #机械臂组件名称
           tgt_pos=np.zeros(3), #目标位置
           tgt_rotmat=np.eye(3), #目标姿态
           seed_jnt_values=None, #种子关节值，用于逆运动学求解的初始关节角度
           max_niter=200, #最大迭代次数
           tcp_jnt_id=None, #工具末端点TCP所在的关节ID
           tcp_loc_pos=None, #工具末端点TCP在指定关节上的局部位置偏移
           tcp_loc_rotmat=None, #工具末端点TCP在指定关节上的局部旋转矩阵偏移
           local_minima: str = "end", #处理局部极小值的方法
           toggle_debug=False):#True则开启调试模式，可能会输出详细的求解过程信息
        #通过self.manipulator_dict[component_name]获取对应的机械臂对象，然后调用该对象的ik方法
        return self.manipulator_dict[component_name].ik(tgt_pos, #目标位置
                                                        tgt_rotmat, #目标姿态
                                                        seed_jnt_values=seed_jnt_values, #种子关节值，用于逆运动学求解的初始关节角度
                                                        max_niter=max_niter, #最大迭代次数
                                                        tcp_jnt_id=tcp_jnt_id, #工具末端点TCP所在的关节ID
                                                        tcp_loc_pos=tcp_loc_pos, #工具末端点TCP在指定关节上的局部位置偏移
                                                        tcp_loc_rotmat=tcp_loc_rotmat,  #工具末端点TCP在指定关节上的局部旋转矩阵偏移
                                                        local_minima=local_minima,   #处理局部极小值的方法
                                                        toggle_debug=toggle_debug) #True则开启调试模式，可能会输出详细的求解过程信息

    def tracik(self,
               component_name: str = "arm",
               urdf_path='',    # 需要提供urdf
               base_link_name='',
               tip_link_name='',
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None):
        return self.manipulator_dict[component_name].tracik(urdf_path=urdf_path,
                                                            base_link_name=base_link_name,
                                                            tip_link_name=tip_link_name,
                                                            tgt_pos=tgt_pos,
                                                            tgt_rotmat=tgt_rotmat,
                                                            seed_jnt_values=seed_jnt_values)

    def manipulability(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       component_name='arm'):
        return self.manipulator_dict[component_name].manipulability(tcp_jnt_id=tcp_jnt_id,
                                                                    tcp_loc_pos=tcp_loc_pos,
                                                                    tcp_loc_rotmat=tcp_loc_rotmat)

    def manipulability_axmat(self,
                             tcp_jnt_id=None,
                             tcp_loc_pos=None,
                             tcp_loc_rotmat=None,
                             component_name='arm',
                             type="translational"):
        return self.manipulator_dict[component_name].manipulability_axmat(tcp_jnt_id=tcp_jnt_id,
                                                                          tcp_loc_pos=tcp_loc_pos,
                                                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                                                          type=type)

    def jacobian(self,
                 component_name='arm',
                 tcp_jnt_id=None,
                 tcp_loc_pos=None,
                 tcp_loc_rotmat=None):
        return self.manipulator_dict[component_name].jacobian(tcp_jnt_id=tcp_jnt_id,
                                                              tcp_loc_pos=tcp_loc_pos,
                                                              tcp_loc_rotmat=tcp_loc_rotmat)

    def rand_conf(self, component_name):
        return self.manipulator_dict[component_name].rand_conf()

    def cvt_conf_to_tcp(self, manipulator_name, jnt_values):
        """
        given jnt_values, this function returns the correspondent global tcp_pos, and tcp_rotmat
        :param manipulator_name:
        :param jnt_values:
        :return:
        author: weiwei
        date: 20210417
        """
        jnt_values_bk = self.get_jnt_values(manipulator_name)
        self.robot_s.fk(manipulator_name, jnt_values)
        gl_tcp_pos, gl_tcp_rotmat = self.robot_s.get_gl_tcp(manipulator_name)
        self.robot_s.fk(manipulator_name, jnt_values_bk)
        return gl_tcp_pos, gl_tcp_rotmat

    def cvt_gl_to_loc_tcp(self, manipulator_name, gl_obj_pos, gl_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_gl_to_loc_tcp(gl_obj_pos, gl_obj_rotmat)

    def cvt_loc_tcp_to_gl(self, manipulator_name, rel_obj_pos, rel_obj_rotmat):
        return self.manipulator_dict[manipulator_name].cvt_loc_tcp_to_gl(rel_obj_pos, rel_obj_rotmat)

    def is_collided(self, obstacle_list=None, otherrobot_list=None, toggle_contact_points=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        输入:
            :param obstacle_list: 障碍物对象
            :param otherrobot_list: 其他机器人对象
            :param toggle_contact_points: debug
        输出:
            CollisionChecker is_collided for details
        author: weiwei
        date: 20201223
        """
        if obstacle_list is None:
            obstacle_list = []
        if otherrobot_list is None:
            otherrobot_list = []
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             otherrobot_list=otherrobot_list,
                                             toggle_contact_points=toggle_contact_points)
        return collision_info

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        raise NotImplementedError

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self_copy.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy
