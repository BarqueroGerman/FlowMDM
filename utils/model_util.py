from model.FlowMDM import FlowMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from model.cfg_sampler import wrap_model
import torch


def load_model(args, device, cond_mode=None, ModelClass=FlowMDM, DiffusionClass=SpacedDiffusion):
    model, diffusion = create_model_and_diffusion(args, cond_mode=cond_mode, ModelClass=ModelClass, DiffusionClass=DiffusionClass)
    model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()
    model = wrap_model(model, args)
    return model, diffusion

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    to_remove = ['t_pos_encoder.pe', 'seqTransEncoder.attn_layers.rotary_pos_emb.last_timestep']
    for tor in to_remove:
        if tor in missing_keys:
            missing_keys.remove(tor)
        if tor in unexpected_keys:
            unexpected_keys.remove(tor)
    #print("WARNING: unexpected keys: {}".format(unexpected_keys))
    return unexpected_keys


def create_model_and_diffusion(args, cond_mode=None, ModelClass=FlowMDM, DiffusionClass=SpacedDiffusion):
    model = ModelClass(**get_model_args(args, cond_mode=cond_mode))
    diffusion = create_gaussian_diffusion(args, DiffusionClass=DiffusionClass)
    return model, diffusion

def get_model_args(args, cond_mode=None):

    # default args
    clip_version = 'ViT-B/32'
    cond_mode = get_cond_mode(args) if cond_mode is None else cond_mode

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'babel':
        data_rep = 'rot6d'
        njoints = 135
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
            'translation': True, 'pose_rep': data_rep, 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': args.num_heads,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 
            'clip_version': clip_version, 'dataset': args.dataset,
            'diffusion_steps': args.diffusion_steps,
            'max_seq_att': args.max_seq_att, 
            # FlowMDM
            'bpe_denoising_step': args.bpe_denoising_step,
            'bpe_training_ratio': args.bpe_training_ratio,
            'rpe_horizon': args.rpe_horizon,
            'use_chunked_att': args.use_chunked_att,
            }


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_vel_rcxyz=args.lambda_vel_rcxyz,
    )

