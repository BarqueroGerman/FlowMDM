import torch
import numpy as np
import argparse
import pickle
import smplx

from utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="common/utils/human_model_files/")
    parser.add_argument("--model_type", type=str, default="smplx", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="demo/results/demo_video/smplx/data.npz")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", type=str, default="demo/results/demo_video/smplx/motion.bvh")
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE",
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    

    names = [
        "Hips",
        "LeftUpLeg",
        "RightUpLeg",
        "Spine",
        "LeftLeg",
        "RightLeg",
        "Spine1",
        "LeftFoot",
        "RightFoot",
        "Spine2",
        "LeftToe",
        "RightToe",
        "Neck",
        "LeftShoulder",
        "RightShoulder",
        "Head",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
        "LeftThumb",
        "RightThumb",
    ]
    names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        'Jaw', 'Left_eye', 'Right_eye']
    
    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:55,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 1
    
    scaling = None
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        # rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        # trans = np.squeeze(poses["trans"], axis=0) # (N, 3)
        rots = poses["poses"] # (N, 24, 3)
        trans = poses["trans"]/5000 # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")
    
    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    positions[:, 0] += trans
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)


# def joints2bvh()

if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
             mirror = args.mirror, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, 
             fps=args.fps, output=args.output)
    
    print("finished!")