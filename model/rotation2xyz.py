# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry


from model.smpl import SMPL, JOINTSTYPE_ROOT, SMPLX
# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices", "smplx"]


class Rotation2xyz:
    def __init__(self, device, include_hands=False, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.include_hands = include_hands
        if include_hands:
            self.smpl_model = SMPLX().eval().to(device)
        else:
            self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, 
                 **kwargs):
        # x: [batch_size, joints, feats, sequence_length]
        # x will be divided in main body, left hand, right hand
        # if glob == True, first rotation is global rotation
        # if translation == True, last position in joints dimension is translation
        if not self.include_hands and x.shape[1] == 23 + int(translation) + int(glob):
            print("Warning: you are using a model without hands, but you are passing hand data (> 23 joints).")
        elif self.include_hands and x.shape[1] == 53 + int(translation) + int(glob):
            print("Warning: you are using a model with hands, but you are not passing hand data (< 53 joints).")

        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        # remove the only non-rotational joint (root)
        if translation:
            x_translations = x[:, -1, :3] # last position in joints dimension is translation
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            f = lambda x, mask: geometry.axis_angle_to_matrix(x[mask])
        elif pose_rep == "rotmat":
            f = lambda x, mask: x[mask].view(-1, x.shape[2], 3, 3)
        elif pose_rep == "rotquat":
            f = lambda x, mask: geometry.quaternion_to_matrix(x[mask])
        elif pose_rep == "rot6d":
            f = lambda x, mask: geometry.rotation_6d_to_matrix(x[mask])
        else:
            raise NotImplementedError("No geometry for this one.")
        rotations = f(x_rotations, mask)

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0] # first rotation is global rotation
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()

        if self.include_hands:
            lh_rotations, rh_rotations = rotations[:, 21:21+15], rotations[:, 36:]
            rotations = rotations[:, :21]
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas, 
                                    left_hand_pose=lh_rotations, right_hand_pose=rh_rotations)
        else:
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
