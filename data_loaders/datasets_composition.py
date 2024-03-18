import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dist_util
import  numpy as np
from data_loaders.tensors import lengths_to_mask
import os
import json
from data_loaders.amass.babel import get_tokens
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.data.dataset import process_tokens
from os.path import join as pjoin

def pad_sample_with_zeros(sample, max_len=250):
    # pad inp, change lenghts, and pad is transition
    seq_len, n_feats = sample.shape
    len_to_pad = max_len - seq_len
    np.zeros_like(sample)
    sample_padding = np.zeros((len_to_pad, n_feats))
    sample = np.concatenate((sample, sample_padding))
    return sample


class CompMDMGeneratedDataset(Dataset):

    def load_model_kwargs_dataset(self, eval_file, scenario=""):
        import json
        with open(eval_file, 'r') as f:
            all_model_kwargs = json.load(f)
            print(f"loaded {eval_file}", len(all_model_kwargs))

            # convert all "lengths" to torch
            final_model_kwargs = []
            for i in range(len(all_model_kwargs)):
                idx = int(all_model_kwargs[i]['id'])
                kwargs = {"y": all_model_kwargs[i]}
                if scenario != "" and scenario is not None and "scenario" in kwargs['y'] and kwargs['y']['scenario'] != scenario:
                    continue # skip this one

                kwargs['y']['lengths'] = torch.tensor([int(v) for v in kwargs['y']['lengths']])
                kwargs['y']['mask'] = lengths_to_mask(kwargs['y']['lengths'], kwargs['y']['lengths'].max()).unsqueeze(1).unsqueeze(1)
                final_model_kwargs.append((idx, kwargs))
            
            assert len(final_model_kwargs) > 0, f"No model kwargs found for this scenario: {scenario}"
            print(f"loaded {len(final_model_kwargs)} model kwargs for scenario {scenario if scenario != '' else '> all <'}")

        return iter(final_model_kwargs)
    
    def process_tokens(self, tokens):
        return process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

    def __init__(self, args, model, diffusion, mm_num_samples, mm_num_repeats, eval_file):

        dataloader = self.load_model_kwargs_dataset(eval_file)
        #assert mm_num_samples < len(dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        generated_motion = []
        mm_generated_motions = []
        num_seqs = 32 # ALWAYS
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(len(dataloader), mm_num_samples // num_seqs +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        #skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        progress=False,
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    } for bs_i in range(num_seqs)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(num_seqs)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'mm_motions': mm_motions[bs_i::num_seqs],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(num_seqs)]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption = data['motion'], data['length'], data['caption']
        
        if self.dataset_name != "babel": # babel already takes care of its de)normalization itself
            normed_motion = motion
            denormed_motion = normed_motion * self.std + self.mean
            renormed_motion = (denormed_motion - self.mean_for_eval) / self.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention
        
        tokens = get_tokens(caption)
        word_embeddings, pos_one_hots, sent_len, tokens = self.process_tokens(tokens)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens

    def inv_transform(self, data):
        return data * self.std_for_eval + self.mean_for_eval
    
    def switch_target(self, target):
        assert target in ['motion', ], "Only motion eval target is available for non-unfolding dataset"

class CompMDMUnfoldingGeneratedDataset(CompMDMGeneratedDataset):

    def __init__(self, args, model, diffusion, max_motion_length, eval_file, w_vectorizer=None, opt=None, precomputed_folder=None, scenario=""):
        self.dataset_name = args.dataset
        self.w_vectorizer = w_vectorizer
        self.opt = opt
        assert self.dataset_name == "babel" or self.dataset_name == "humanml", "Only babel and humanml are supported"
        if self.dataset_name == "humanml":
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        dataloader = self.load_model_kwargs_dataset(eval_file, scenario=scenario)
        self.max_motion_length = max_motion_length

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        generated_transitions = []
        generated_motion = []

        os.makedirs(precomputed_folder, exist_ok=True)
        with torch.no_grad():
            for i, model_kwargs in tqdm(dataloader):
                precomputed_file_kwargs = os.path.join(precomputed_folder, f'{i:02d}_kwargs.json')
                precomputed_file_pt = os.path.join(precomputed_folder, f'{i:02d}.pt')
                precomputed_file_npy = os.path.join(precomputed_folder, f'{i:02d}.npy')
                if os.path.exists(precomputed_file_kwargs):
                    # load it from precomputed file
                    #print(f'Loading precomputed file {precomputed_file}')
                    loaded_kwargs = json.load(open(precomputed_file_kwargs, 'r'))
                    kwargs = loaded_kwargs if "y" in loaded_kwargs else {"y": loaded_kwargs} # to keep compatibility with the old format
                    # assert equal
                    assert kwargs['y']['lengths'] == model_kwargs['y']['lengths'].tolist()
                    assert kwargs['y']['text'] == model_kwargs['y']['text']

                    if os.path.exists(precomputed_file_pt):
                        unfolded = torch.load(precomputed_file_pt)
                    elif os.path.exists(precomputed_file_npy):
                        # if unfolded contains rots and transl, we need to convert them to our features representation
                        unfolded = np.load(precomputed_file_npy, allow_pickle=True).item()
                        rots = torch.tensor(unfolded["rots"]) # --> [seq_len, 22, 3, 3], rot matrices
                        transl = torch.tensor(unfolded["transl"]) # --> [seq_len, 3], translations

                        # we need to go from 3x3 matrices to axis angle, and then we got it :)
                        from utils.rotation_conversions import matrix_to_axis_angle
                        rots = matrix_to_axis_angle(rots) # --> [seq_len, 22, 3], axis angle

                        from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                        smpl_data = {
                            "poses": rots,
                            "trans": transl,
                        }
                        smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
                        from data_loaders.amass.transforms import SlimSMPLTransform
                        transform = SlimSMPLTransform(batch_size=32, name='SlimSMPLTransform', ename='smplnh', normalization=True)
                        features = transform.rots2rfeats(smpl_data)
                        unfolded = features.permute(1, 0).unsqueeze(1).unsqueeze(0)
                        torch.save(unfolded, precomputed_file_pt)
                    else:
                        assert False, "Precomputed file not found"
                    
                else:
                    assert model is not None and diffusion is not None, "Model and diffusion must be provided for evaluation if precomputed files are not available"
                    # compute it
                    model.eval()
                    unfolded = diffusion.p_sample_loop(
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False#True,
                    ) # --> [1, 263, 1, BS * single_lengths], where single_lengths is each length in model_kwargs['y']['lengths']

                    # store it
                    #print(f'Storing precomputed file {precomputed_file}')
                    torch.save(unfolded, precomputed_file_pt)
                    # save in float32
                    json.dump({'lengths': model_kwargs['y']['lengths'].tolist(), 'text': model_kwargs['y']['text']}, open(precomputed_file_kwargs, 'w'))


                # evaluates on the whole motion segment
                start = 0
                num_seqs = len(model_kwargs['y']['lengths'])
                for bs_i in range(num_seqs):
                    end = (start + model_kwargs['y']['lengths'][bs_i]) if bs_i != num_seqs - 1 else None
                    motion_slice = unfolded[..., start:end].squeeze().permute(1, 0).cpu().numpy()
                    assert motion_slice.shape[0] == model_kwargs['y']['lengths'][bs_i], f'{motion_slice.shape[0]} != {model_kwargs["y"]["lengths"][bs_i]}'

                    generated_motion.append({
                        'motion': pad_sample_with_zeros(motion_slice, self.max_motion_length),
                        'length': model_kwargs['y']['lengths'][bs_i],
                        'caption': model_kwargs['y']['text'][bs_i],
                    })
                    start = end

                # only keeps the transition
                l_margin = (args.transition_length // 2)
                r_margin = args.transition_length - l_margin
                mid = 0
                for bs_i in range(num_seqs - 1): # last one has no transition
                    mid += model_kwargs['y']['lengths'][bs_i]
                    motion_slice = unfolded[..., mid - l_margin : mid + r_margin].squeeze().permute(1, 0).cpu().numpy()
                    assert motion_slice.shape[0] == args.transition_length

                    generated_transitions.append({
                        'motion': motion_slice, # all transitions with length args.transition_length, so no need for padding
                        'length': args.transition_length,
                        'caption': model_kwargs['y']['text'][bs_i],
                    })

        self.generated_inbetween = generated_motion
        self.generated_transitions = generated_transitions
        
        self.switch_target('motion') # 'motion' or 'transition'

    def switch_target(self, target):
        """
        Switches between 'motion' and 'transition' targets. In 'motion' target, the dataset returns the full motion segment. 
        In 'transition' target, the dataset returns only the transition part of the motion
        """
        assert target in ['motion', 'transition']
        self.target = target
        self.generated_motion = self.generated_inbetween if self.target == 'motion' else self.generated_transitions

