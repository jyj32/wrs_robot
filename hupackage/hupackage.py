import basis.robot_math as rm
import modeling.geometric_model as gm
import numpy as np
import hu.humath as hm
def debugpos(pos, rot, base):
    gm.gen_frame(pos, rot).attach_to(base)


def normal_from_3point(p1, p2, p3):
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]
    x3, y3, z3 = p3[0], p3[1], p3[2]
    # print(x3, y3, z3)
    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return rm.unit_vector(np.array([a, b, c]))



def find_circle_center_3d(p1,p2,p3):
    normal = normal_from_3point(p1,p2,p3)
    a = (p2 + p1)/2
    b = (p3 + p1)/2
    a_dir = rm.unit_vector(p2-p1)
    b_dir = rm.unit_vector(p3-p1)
    rot_1 = rm.rotmat_from_axangle(normal, np.pi/2)
    rot_2 = rm.rotmat_from_axangle(normal, -np.pi/2)
    # a_ver = np.dot(rot_1, a_dir)
    # b_ver = np.dot(rot_2, b_dir)
    s_1 = hm.getsurfaceequation(hm.getsurfacefrom3pnt([p1, p2, p3]),p1)
    s_2 = hm.getsurfaceequation(a_dir, a)
    s_3 = hm.getsurfaceequation(b_dir, b)
    # print(s_1)
    # print(s_2)
    # print(s_3)

    A = np.array([s_1[:3],s_2[:3],s_3[:3]])
    B = np.array([s_1[3], s_2[3], s_3[3]])
    r = np.linalg.solve(A, -B)
    print(r)
    return r
# 输入三个点的坐标
# p1 = np.array([0,0,0])
# p2 = np.array([2,0,0])
# p3 = np.array([0,2,0])
# find_circle_center_3d(p1, p2, p3)
# print(f"外接球球心坐标：({center_x}, {center_y}, {center_z})")
