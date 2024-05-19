# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion_mix
import json
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper

from visualization.joints2bvh import Joint2BVHConvertor

converter = Joint2BVHConvertor()

datasets_fps = {
    "humanml": 20,
    "babel": 30
}

def feats_to_xyz(sample, dataset, batch_size=1):
    if dataset == 'humanml': # for HumanML3D
        n_joints = 22
        mean = np.load('dataset/HML_Mean_Gen.npy')
        std = np.load('dataset/HML_Std_Gen.npy')
        sample = sample.cpu().permute(0, 2, 3, 1)
        sample = (sample * std + mean).float()
        sample = recover_from_ric(sample, n_joints) # --> [1, 1, seqlen, njoints, 3]
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # --> [1, njoints, 3, seqlen]
    elif dataset == 'babel': # [bs, 135, 1, seq_len] --> 6 * 22 + 3 for trajectory
        from data_loaders.amass.transforms import SlimSMPLTransform
        transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        all_feature = sample #[bs, nfeats, 1, seq_len]
        all_feature_squeeze = all_feature.squeeze(2) #[bs, nfeats, seq_len]
        all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) #[bs, seq_len, nfeats]
        splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
        sample_list = []
        for seq in splitted[0]:
            all_features = seq
            Datastruct = transform.SlimDatastruct
            datastruct = Datastruct(features=all_features)
            sample = datastruct.joints

            sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
        sample = torch.cat(sample_list)
    else:
        raise NotImplementedError("'feats_to_xyz' not implemented for this dataset")
    return sample

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = datasets_fps[args.dataset]
    assert args.instructions_file == '' or 'json' == args.instructions_file.split('.')[-1], "Instructions file must be a json file"
    dist_util.setup_dist(args.device)
    if out_path == '': # if unspecified, save in the same folder as the model
        out_path = os.path.join(os.path.dirname(args.model_path),
                                '{}_s{}'.format(niter, args.seed))
        if args.instructions_file != '':
            out_path += '_' + os.path.basename(args.instructions_file).replace('.json', '').replace(' ', '_').replace('.', '')

    animation_out_path = out_path
    os.makedirs(animation_out_path, exist_ok=True)

    # ================= Load texts + lengths and adapt batch size ================
    # this block must be called BEFORE the dataset is loaded
    is_using_data = args.instructions_file == ''
    if not is_using_data: 
        assert os.path.exists(args.instructions_file)
        # load json
        with open(args.instructions_file, 'r') as f:
            instructions = json.load(f)
            assert "text" in instructions and "lengths" in instructions, "Instructions file must contain 'text' and 'lengths' keys"
            assert len(instructions["text"]) == len(instructions["lengths"]), "Instructions file must contain the same number of 'text' and 'lengths' elements"
        num_instructions = len(instructions["text"])
        args.batch_size = num_instructions
        args.num_samples = 1
    else:
        num_instructions = args.num_samples
        args.batch_size = num_instructions
        args.num_samples = 1

    # ================= Load dataset or prepare model_kwargs for inference ================
    if is_using_data:
        print('Loading dataset...')
        if args.split == "test" and args.dataset == "babel":
            args.split = "val" # Babel does not have a test set

        try:
            data = load_dataset(args, args.split)
        except Exception as e:
            print(f'Error while loading dataset: {e}')
            return
        
        if is_using_data:
            iterator = iter(data)
            sample_gt, model_kwargs = next(iterator)
            
        j = { "sequence": [] }
        for i in range(num_instructions):
            length = model_kwargs['y']['lengths'][i].item()
            text = model_kwargs['y']['text'][i]
            j["sequence"].append([length, text])
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(j, f)

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    else: # from instructions
        json_lengths = instructions["lengths"]
        json_texts = instructions["text"]
        mask = torch.ones((len(json_texts), max(json_lengths)))
        for i, length in enumerate(json_lengths):
            mask[i, length:] = 0
        model_kwargs = {'y': {
            'mask': mask,
            'lengths': torch.tensor(json_lengths),
            'text': list(json_texts),
            'tokens': [''],
        }}
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(instructions, f)
            
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    print(list(zip(list(model_kwargs['y']['text']), list(model_kwargs['y']['lengths'].cpu().numpy()))))

    # ================= Load model and diffusion wrapper ================
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    # ================= Sample ================
    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetition #{rep_i}]')
        sample = diffusion.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )
        sample = feats_to_xyz(sample, args.dataset)

        c_text = ""
        for i in range(num_instructions):
            c_text += model_kwargs['y']['text'][i] + " /// "

        all_text.append(c_text)
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].sum().unsqueeze(0))#.cpu().numpy())

        print(f"created {rep_i+1}/{args.num_repetitions} human motion compositions.")

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # ================= Save results + visualizations ================
    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain
    
    sample_print_template, row_print_template, \
    sample_file_template, row_file_template = construct_template_variables(args.unconstrained)

    try:
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.num_samples]
            motion = all_motions[rep_i*args.num_samples].transpose(2, 0, 1)
            save_file = sample_file_template.format(rep_i)
            print(sample_print_template.format(rep_i, save_file))
            animation_save_path = os.path.join(animation_out_path, save_file)
            # saving motion as BVH
            bvh_path = os.path.join(animation_out_path, "motion.bvh")
            _, _ = converter.convert(motion, filename=bvh_path, iterations=1000, foot_ik=False)
            lengths_list = model_kwargs['y']['lengths']
            captions_list = []
            for c, l in zip(caption.split(" /// "), lengths_list):
                captions_list += [c,] * l
            plot_3d_motion_mix(animation_save_path, skeleton, motion, dataset=args.dataset, title=captions_list, fps=fps,
                        vis_mode='alternate', lengths=lengths_list)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
    except Exception as e:
        print(f'Error while processing sample: {e}')

    save_multiple_samples(args, animation_out_path,
                                            row_print_template, row_file_template,
                                            caption, rep_files)

    abs_path = os.path.abspath(animation_out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, row_file_template, caption, rep_files):
    all_rep_save_file = row_file_template
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(all_rep_save_file))


def construct_template_variables(unconstrained):
    row_file_template = 'sample_all.mp4'
    if unconstrained:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'
    else:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[Rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'

    return sample_print_template, row_print_template, \
           sample_file_template, row_file_template


def load_dataset(args, split):
    n_frames = 150 # this comes from PriorMDM, so I'm using it here as well
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames

    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split=split,#split,
                              load_mode='gen',#'eval',
                              protocol=args.protocol,
                              pose_rep=args.pose_rep,
                              num_workers=1)
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()