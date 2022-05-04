import numpy as np
from scipy.spatial.transform import Rotation as R


def Rot_z(angle):
    return np.array([   [np.cos(angle),-np.sin(angle),0],
                        [np.sin(angle),np.cos(angle),0],
                        [0,0,1]])
def Rot_y(angle):
    return np.array([   [np.cos(angle),0, np.sin(angle)],
                        [0,1,0],
                        [-np.sin(angle),0, np.cos(angle)]])


def Rot_x(angle):
    return np.array([   [1,0,0],
                        [0, np.cos(angle),-np.sin(angle)],
                        [0, np.sin(angle),np.cos(angle)]])


def get_pos_and_quat_from_trans_matrix(Trans_matrix, rbot_base_frame, conversion):

    """ This function converts Transformation matrix from physical system to correct camera pose and
        camera quaternion in camera frame in mujoco 

    Args:
        Trans_matrix (4x4 array): Transformation matrix from robot base to camera frame in the lab, described in NED

        rbot_base_frame (1x3 array): Location of robot base frame in simulator world

        conversion (bool): Either "x-y-z" converions or "temporary" untill right camera calibration

    Returns: cam_pose (1x3 array), cam_quat (1x4 array)    
    """
    Ned_to_Enu_conversion_x_neg_y_neg_z = np.array([[1,0,0],
                                                    [0,-1,0],
                                                    [0,0,-1]
    ])

    cam_pose = np.add(Trans_matrix[:3,3]/1000, rbot_base_frame)

    if conversion:
        rotation_matrix_enu = Trans_matrix[:3,:3]@Ned_to_Enu_conversion_x_neg_y_neg_z
        
    else:
        cam_pose = np.array([1.23, 0, 1.25]) # Original [1.53, 0, 1.45]
        rotation_matrix_enu = Rot_z(np.pi/2)@Rot_x(np.pi/3) # Original Rot_z(np.pi/2)@Rot_x(np.pi/2.65)
    
    cam_quat = R.from_matrix(rotation_matrix_enu).as_quat()[[3, 0, 1, 2]]
    return cam_pose, cam_quat

def adjust_width_of_image(height: int):
    ratio = 1944/1200
    dif = 0.00001
    assert height*ratio % 1 <= dif, "Make height pixel divisible by 1944 and 1200"
    return int(height*ratio)




