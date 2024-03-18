
import torch
import clip
import torch.nn as nn
from model.MDM import PositionalEncoding, TimestepEmbedder


class TextConditionalModel(nn.Module):
    def __init__(self, latent_dim=256, cond_mode="no_cond", cond_mask_prob=0., dropout=0.0, clip_dim=512, clip_version=None, **kargs):
        super().__init__()
        self.cond_mode = cond_mode
        assert self.cond_mode in ["no_cond", "text"]
        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        
        if cond_mode != 'no_cond':
            if 'text' in cond_mode:
                self.embed_text = nn.Linear(clip_dim, latent_dim)
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            else:
                raise NotImplementedError("only conditioning with text is implemented for now")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def compute_embedding(self, x, timesteps, y):
        """
            Explanation on what the buffers do:
            - emb: stores the embedding for the current condition. It is used to avoid recomputing the embedding if the condition is the same (big inference speedup)
            - emb_hash: stores the hash of the condition. It is used to check if the condition is the same as the one stored in emb
            - emb_forcemask: stores the embedding for the current condition, but with the mask forced to True. It is used to avoid recomputing the embedding for the unconditional case
            - emb_forcemask_hash: stores the hash of the condition. It is used to check if the condition is the same as the one stored in emb_forcemask
        """
        bs, njoints, nfeats, nframes = x.shape

        multitext_mode = "all_texts" in y or not isinstance(y['text'][0], str)
        key = "all_texts" if "all_texts" in y else "text"

        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get('uncond', False)
        if not force_mask:
            if 'text' == self.cond_mode:
                primitive = frozenset(y[key]) if not multitext_mode else frozenset((frozenset(txts) for txts in y[key]))
            else:
                raise ValueError
            
            hash_value = hash(primitive)
            recompute = not hasattr(self, 'emb_hash') or self.emb_hash != hash_value
            if not recompute:
                return time_emb + self.emb
        else:
            hash_value = hash(frozenset(x.shape))
            recompute = not hasattr(self, 'emb_forcemask_hash') or self.emb_forcemask_hash != hash_value
            if not recompute:
                return time_emb + self.emb_forcemask

        # compute embedding
        if not multitext_mode: # --> single text training (e.g. HumanML3D dataset) / inference
            enc_text = self.encode_text(y['text']) if "text_embeddings" not in y else y["text_embeddings"] # if precomputed --> faster
            cond_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            cond_emb = cond_emb.unsqueeze(0).expand(nframes, -1, -1) # [T, N, d]
        else: # --> multi-text training / inference (e.g. Babel dataset)
            if "text_embeddings" in y: # preloaded for fast training / eval
                enc_text = y["text_embeddings"]
            else:
                # 'conditions_mask' has shape [I, T, N] where I is the number of different conditions, N is batch size, T is sequence length.
                # y[key] is a list of size I with each element being a list of strings of size N
                # We need to encode the text and build the embedding matrix
                texts_list = y[key]
                # homogeneize all lists to same length to stack them later
                max_len = max([len(texts) for texts in texts_list])
                for i, texts in enumerate(texts_list):
                    if len(texts) < max_len:
                        texts_list[i] = texts + [''] * (max_len - len(texts))
                enc_text = [self.encode_text(text) for text in texts_list]
                enc_text = torch.stack(enc_text, dim=1)

            I, N, d = enc_text.shape
            enc_text = enc_text.reshape(-1, enc_text.shape[-1]) # [I*N, d]
            embedded_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)).reshape(I, N, d) # [I, N, d]

            conditions_mask = y['conditions_mask'] # [I, T, N]
            conditions_mask = conditions_mask.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim) # [I, T, N, d]
            cond_emb = torch.zeros(conditions_mask.shape[1:], device=embedded_text.device) # [T, N, d]
            for i in range(I):
                m = conditions_mask[i] # [T, N, d]
                cond_emb = cond_emb + m * embedded_text[i].unsqueeze(0) # [T, N, d] --> [T, N, d]
        
        # send to buffer
        if force_mask:
            self.register_buffer('emb_forcemask', cond_emb, persistent=False)
            self.register_buffer('emb_forcemask_hash', torch.tensor(hash(frozenset(x.shape))), persistent=False)
        else:
            self.register_buffer('emb', cond_emb, persistent=False)
            self.register_buffer('emb_hash', torch.tensor(hash(primitive)), persistent=False)

        return time_emb + cond_emb
