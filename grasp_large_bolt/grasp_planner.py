import os
import numpy as np
import pickle

import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as wd
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.gripper_interface as gp
import robot_sim.end_effectors.gripper.dh50.dh50 as hnd


class GraspPlanner:
    def __init__(self, obj_name: str, gripper_sim: gp.GripperInterface):
        self.obj_name = obj_name
        self.obj = cm.CollisionModel(f'./object/{obj_name}.stl')
        self.gripper_s = gripper_sim
        self.grasp_info_list = []

    def plan_grasp(self, load=False, manual_set = True):
        if manual_set:
            grasp_info_list = []
            rot_list = [rm.rotmat_from_axangle([0, 0, 1], np.radians(angle)) for angle in [0, -60, 60, -120, 120, -180 ]]
            hand_center_pos = np.array([0, 0, -0.001])
            for rot in rot_list:
                grasp_info_list.append([0.05, hand_center_pos, rot, hand_center_pos, rot])
            self.grasp_info_list = grasp_info_list
            return self.grasp_info_list

        debug_data_path = f'./debug_data/{self.obj_name}_grasp_info.pkl'
        if load and os.path.exists(debug_data_path):
            with open(debug_data_path, 'rb') as f:
                self.grasp_info_list = pickle.load(f)
        else:
            self.grasp_info_list = gpa.plan_grasps(self.gripper_s, self.obj,
                                                   angle_between_contact_normals=np.radians(170),
                                                   openning_direction='loc_x',
                                                   rotation_interval=np.radians(90),
                                                   max_samples=10, min_dist_between_sampled_contact_points=.001,
                                                   contact_offset=.002)
            with open(debug_data_path, 'wb') as f:
                pickle.dump(self.grasp_info_list, f)
        return self.grasp_info_list

    def visualization(self, base:wd.World, rgba=None):
        self.obj.attach_to(base)
        for grasp_info in self.grasp_info_list:
            self.gripper_s.grip_at_with_jcpose(grasp_info[1], grasp_info[2], grasp_info[0])
            self.gripper_s.gen_meshmodel(rgba=rgba).attach_to(base)

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    gripper_s = hnd.Dh50()

    grasp_planner = GraspPlanner(obj_name='M16_50', gripper_sim=gripper_s)
    grasp_info = grasp_planner.plan_grasp(manual_set=True)
    grasp_planner.visualization(base, [0, 1, 0, 0.7])

    base.run()