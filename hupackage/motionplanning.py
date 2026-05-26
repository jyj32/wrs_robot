import math
import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp

class PrimaryPlanner(object):
    def __init__(self, robot_s):
        '''

        :param robot_s:
        '''
        self.robot_s = robot_s
        self.inik_slvr = inik.IncrementalNIK(self.robot_s)
        self.rrtc_planner = rrtc.RRTConnect(self.robot_s)

    def gen_linear_moveto(self,
                          component_name,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          goal_tcp_pos,
                          goal_tcp_rotmat,
                          obstacle_list,
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_debug=False):
        return self.inik_slvr.gen_linear_motion(
                                      component_name,
                                      start_tcp_pos,
                                      start_tcp_rotmat,
                                      goal_tcp_pos,
                                      goal_tcp_rotmat,
                                      obstacle_list,
                                      granularity=granularity,
                                      seed_jnt_values=seed_jnt_values,
                                      toggle_debug=toggle_debug)

    def gen_linear_move_obj_to(self,
                          component_name,
                          start_obj_pos,
                          start_obj_rotmat,
                          goal_obj_pos,
                          goal_obj_rotmat,
                          grasp_info_list,
                          obstacle_list,
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_debug=False):
        start_tcp_pos
        return self.inik_slvr.gen_linear_motion(
                                      component_name,
                                      start_tcp_pos,
                                      start_tcp_rotmat,
                                      goal_tcp_pos,
                                      goal_tcp_rotmat,
                                      obstacle_list,
                                      granularity=granularity,
                                      seed_jnt_values=seed_jnt_values,
                                      toggle_debug=toggle_debug)