import numpy as np
import torch
import torch.nn as nn
from model.rotation2xyz import Rotation2xyz
from model.MDM import InputProcess, OutputProcess
from model.base_models import TextConditionalModel
from model.x_transformers.x_transformers import ContinuousTransformerWrapper, Encoder


class BPE_Schedule():
    def __init__(self, training_rate: float, inference_step: int, max_steps: int) -> None:
        assert training_rate >= 0 and training_rate <= 1, "training_rate must be between 0 and 1"
        assert inference_step == -1 or (inference_step >= 0 and inference_step <= max_steps), "inference_step must be between 0 and max_steps"
        self.training_rate = training_rate
        self.inference_step = inference_step
        self.max_steps = max_steps
        self.last_random = None

    def step(self, t: torch.Tensor, training: bool):
        self.last_random = torch.rand(t.shape[0], device=t.device)

    def get_schedule_fn(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # False --> absolute
        # True --> relative
        if training: # at TRAINING: then random dropout
            return self.last_random < self.training_rate
        # at INFERENCE: step function as BPE schedule
        elif self.inference_step == -1: # --> all denoising chain with APE (absolute)
            return torch.zeros_like(t, dtype=torch.bool)
        elif self.inference_step == 0: # --> all denoising chain with RPE (relative)
            return torch.ones_like(t, dtype=torch.bool)
        else: # --> BPE with binary step function. Step from APE to RPE at "self.inference_step"
            return ~(t > self.max_steps - self.inference_step)
    
    def use_bias(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # function that returns True if we should use the absolute bias (only when using multi-segments **inference**)
        assert (t[0] == t).all(), "Bias from mixed schedule only supported when using same timestep for all batch elements: " + str(t)
        return ~self.get_schedule_fn(t[0], training) # if APE --> use bias to limit attention to the each subsequence

    def get_time_weights(self, t: torch.Tensor, training: bool) -> torch.Tensor:
        # 0 --> absolute
        # 1 --> relative
        return self.get_schedule_fn(t, training).to(torch.int32)
    

class FlowMDM(TextConditionalModel):
    def __init__(self, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 data_rep='rot6d', dataset='babel', 
                 clip_dim=512, clip_version=None, cond_mode="no_cond", cond_mask_prob=0.,
                 **kargs):
        super().__init__(latent_dim=latent_dim, cond_mode=cond_mode, cond_mask_prob=cond_mask_prob, dropout=dropout, clip_dim=clip_dim, clip_version=clip_version)
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_feats = self.njoints * self.nfeats
        self.max_seq_att = kargs.get('max_seq_att', 1024)
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.process_cond_input = [nn.Linear(2*self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]

        print(f"FlowMDM init")
        self.use_chunked_att = kargs.get('use_chunked_att', False)
        bpe_training_rate = kargs.get('bpe_training_ratio', 0.5) # for training, we dropout with prob 50% --> APE vs RPE
        bpe_inference_step = kargs.get('bpe_denoising_step', None)
        diffusion_steps = kargs.get('diffusion_steps', None)
        self.bpe_schedule = BPE_Schedule(bpe_training_rate, bpe_inference_step, diffusion_steps)
        ws = kargs.get('rpe_horizon', -1) # Max attention horizon
        self.local_attn_window_size = 200 if ws == -1 else ws
        print("[Training] RPE/APE rate:", bpe_training_rate)
        print(f"[Inference] BPE switch from APE to RPE at denoising step {bpe_inference_step}/{diffusion_steps}.")
        print("Local attention window size:", self.local_attn_window_size)

        self.seqTransEncoder = ContinuousTransformerWrapper(
            dim_in = self.latent_dim, dim_out = self.latent_dim,
            emb_dropout = self.dropout,
            max_seq_len = self.max_seq_att,
            use_abs_pos_emb = True,
            absolute_bpe_schedule = self.bpe_schedule, # bpe schedule for absolute embeddings (APE)
            attn_layers = Encoder(
                dim = self.latent_dim,
                depth = self.num_layers,
                heads = self.num_heads,
                ff_mult = int(np.round(self.ff_size / self.latent_dim)), # 2 for MDM hyper params
                layer_dropout = self.dropout, cross_attn_tokens_dropout = 0,

                # ======== FLOWMDM ========
                custom_layers=('A', 'f'), # A --> PCCAT
                custom_query_fn = self.process_cond_input, # function that merges the condition into the query --> PCCAT dense layer (see Fig. 3)
                attn_max_attend_past = self.local_attn_window_size,
                attn_max_attend_future = self.local_attn_window_size,
                # ======== RELATIVE POSITIONAL EMBEDDINGS ========
                rotary_pos_emb = True, # rotary embeddings
                rotary_bpe_schedule = self.bpe_schedule, # bpe schedule for rotary embeddings (RPE)
            )
        )

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def forward(self, x, timesteps, y):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        inside y: model_kwargs with mask, pe_bias, pos_pe_abs, conditions_mask. See DiffusionWrapper_FlowMDM.
        """
        bs, njoints, nfeats, nframes = x.shape
        mask = (y['mask'].reshape((bs, nframes))[:, :nframes].to(x.device)).bool() # [bs, max_frames]

        self.bpe_schedule.step(timesteps, self.training) # update the BPE scheduler (decides either APE or RPE for each timestep)
        if self.training or self.bpe_schedule.use_bias(timesteps, self.training):
            pe_bias = y.get("pe_bias", None) # This is for limiting the attention to inside each conditioned subsequence. The BPE will decide if we use it or not depending on the dropout at training time.
            chunked_attn = False
        else: # when using RPE at inference --> we use the bias to limit the attention to the each subsequence
            pe_bias = None
            chunked_attn = self.use_chunked_att # faster attention for inference with RPE for very long sequences (see LongFormer paper for details)

        # store info needed for the relative PE --> rotary embedding
        rotary_kwargs = {'timesteps': timesteps, 'pos_pe_abs': y.get("pos_pe_abs", None), 'training': self.training, 'pe_bias': pe_bias }

        # ============== INPUT PROCESSING ==============
        emb = self.compute_embedding(x, timesteps, y)
        x = self.input_process(x) # [seqlen, bs, d]

        # ============== MAIN ARCHITECTURE ==============
        # APE or RPE is injected inside seqTransEncoder forward function
        x, emb = x.permute(1, 0, 2), emb.permute(1, 0, 2)
        output = self.seqTransEncoder(x, mask=mask, cond_tokens=emb, attn_bias=pe_bias, rotary_kwargs=rotary_kwargs, chunked_attn=chunked_attn)  # [bs, seqlen, d]
        output = output.permute(1, 0, 2)  # [seqlen, bs, d]

        # ============== OUTPUT PROCESSING ==============
        return self.output_process(output)  # [bs, njoints, nfeats, nframes]


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

