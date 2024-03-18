
import torch
import torch as th
from copy import deepcopy


class DiffusionWrapper_FlowMDM():
    def __init__(self, args, diffusion, model):
        self.model = model
        self.diffusion = diffusion
        self.guidance_param = args.guidance_param

    def add_bias_and_absolute_matrices(self, model_kwargs, shape, device):
        """
        We build:
        > pe_bias --> [T, T] matrix with -inf and 0's limiting where the attention during APE mode focuses (0's), i.e., inside each subsequence
        > pos_pe_abs --> [T] matrix with the absolute position of each frame in each subsequence (for injecting the APE sinusoidal correctly during APE mode).
        """
        nframes = shape[-1]

        pos_pe_abs = torch.zeros((nframes, ), device=device, dtype=torch.float32)
        pe_bias = torch.full((nframes, nframes), float('-inf'), device=device, dtype=torch.float32)

        s = 0 # start
        for length in model_kwargs['y']['lengths']:
            pos_pe_abs[s:s+length] = torch.arange(length, device=device, dtype=torch.float32)
            pe_bias[s:s+length, s:s+length] = 0 # only attend to the segment for the absolute modeling part of the schedule
            s += length

        model_kwargs['y']['pe_bias'] = pe_bias # in FlowMDM forward, it is selected according to the BPE schedule if active
        model_kwargs['y']['pos_pe_abs'] = pos_pe_abs.unsqueeze(0) # needs batch size

    def add_conditions_mask(self, model_kwargs, num_frames, device):
        """
        We build a mask of shape [S, T, 1] where S is the number of motion subsequences, T is the max. sequence length.
        For each subsequence, the mask is True only for the frames that belong to the subsequence.
        """
        num_samples = len(model_kwargs["y"]["lengths"])
        conditions_mask = th.zeros((num_samples, num_frames, 1), device=device, dtype=th.bool)
        s = 0
        MARGIN = 0
        for i, length in enumerate(model_kwargs["y"]["lengths"]):
            conditions_mask[i, s+MARGIN:s+length-MARGIN, :] = True # all batch elements have the same instructions
            s += length
        model_kwargs['y']['conditions_mask'] = conditions_mask

    def p_sample_loop(
        self,
        model_kwargs=None, # list of dicts
        **kwargs,
    ):
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model_kwargs=model_kwargs,
            **kwargs,
        )):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        noise=None,
        model_kwargs=None, # list of dicts
        device=None,
        progress=False,
        **kwargs,
    ):
        bs, nframes = 1, model_kwargs['y']['lengths'].sum().item()
        shape = (bs, self.model.njoints, self.model.nfeats, nframes) # all batch elements form the same sequence

        if device is None:
            device = next(self.model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        model_kwargs = deepcopy(model_kwargs)
        self.add_conditions_mask(model_kwargs, nframes, device)
        self.add_bias_and_absolute_matrices(model_kwargs, shape, device)
        model_kwargs["y"]["mask"] = th.ones((bs, nframes), device=device, dtype=th.bool)
        model_kwargs["y"]["lengths"] = th.tensor([nframes], device=device, dtype=th.int64)
        model_kwargs["y"]["scale"] = th.ones(bs, device=device) * self.guidance_param
        # texts are joined as well
        model_kwargs["y"]["all_texts"] = [model_kwargs["y"]["text"], ]
        model_kwargs["y"]["all_lengths"] = [model_kwargs["y"]["lengths"], ]
        model_kwargs["y"]["text"] = " -- ".join(model_kwargs["y"]["text"])

        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for t in indices:
                
            with th.no_grad():

                t = th.tensor([t] * shape[0], device=device)
                out = self.diffusion.p_sample(
                    self.model,
                    img,
                    t,
                    model_kwargs=model_kwargs,
                    **kwargs,
                )

                yield out
                img = out["sample"]
