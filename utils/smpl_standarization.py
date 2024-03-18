import numpy as np
import torch
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_euler_angles
from scipy.spatial.transform import Rotation as R


def rotate_smplh(data, degrees, axis):
    # original thetas, degrees in º. axis in [x, y, z]
    global_rot = axis_angle_to_matrix(torch.tensor(data['global_orient'])).numpy()
    trans_matrix = R.from_euler(axis, degrees, degrees=True).as_matrix()
    new_rots = []
    new_trans = []
    for i in range(global_rot.shape[0]):
        new_rots.append(np.dot(global_rot[i], trans_matrix))
        new_trans.append(np.dot(trans_matrix, data['transl'][i]))
    new_global_rot = np.stack(new_rots, axis=0)
    data['transl'] = np.stack(new_trans, axis=0)
    data['global_orient'] = matrix_to_axis_angle(torch.tensor(new_global_rot)).numpy()


def preprocess_smplh(data, pose_rep):
    # if smplx is passed, then 'jaw_pose', 'leye_pose', 'reye_pose' are also present but ignored because missing in toconcat below.

    # ==== Build motion data ====
    toconcat = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl'] # to keep the order for rotation2xyz.py. 
    # STEP 2: Put on floor (substract y minimum position)
    # STEP 3: XZ at origin (substract xz position)
    data["transl"][:, [0,2]] -= data["transl"][0][[0, 2]]
    # STEP 4: All initially face Z+ (global orientation to Z+) --> Rotation needs to be applied to translation too
    #data['global_orient'] -= data['global_orient'][0] # NOTE: this would work if there was no translation. But there is translation, so we need to rotate it too.
    initial_rot = axis_angle_to_matrix(torch.tensor(data['global_orient'][0]))
    euler_angles = matrix_to_euler_angles(initial_rot, "XYZ")
    angles_degrees = np.rad2deg(euler_angles)
    rotate_smplh(data, -angles_degrees, "xyz")

    # STEP 5: apply transformation to motion rep
    for k in toconcat:
        if pose_rep == "rot6d":
            from utils.rotation_conversions import axis_angle_to_rotation_6d
            if k == "transl":
                # pad translation with zeros in 3 channels (no rotational meaning)
                data[k] = np.concatenate([data[k], np.zeros((data[k].shape[0], 3))], axis=1)[:, np.newaxis, :] # motion -> (n_frames, 1, 3/6)
            else:
                aux = data[k].reshape(data[k].shape[0], -1, 3)
                data[k] = axis_angle_to_rotation_6d(torch.tensor(aux)).numpy() # motion -> (n_frames, n_components/joints, 3/6)
        elif pose_rep == "rotvec": # rotvec --> axis angle default SMPL representation
                data[k] = data[k].reshape(data[k].shape[0], -1, 3) 
        else:
            raise NotImplementedError
    # build motion and reshape to unsqueeze joints dimension
    motion = np.concatenate([data[i] for i in toconcat], axis=1) # motion -> (n_frames, n_components/joints, 3/6)
    return motion