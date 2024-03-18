# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import pickle
import json
from operator import itemgetter
import os
from glob import glob
from platform import architecture
from re import A
from typing import Dict, List, Optional, Tuple
import logging
import joblib

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import random


from .file_io import read_json
from .nlp_consts import fix_spell
from .transforms import Transform

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]
EXCLUDED_ACTIONS = ['transition']
EXCLUDED_ACTIONS_WO_TR = []

def standarize_text(text):
    if text in ["t-pose", "t pose", "tpose"]:
        return "t-pose"
    if text in ["a-pose", "a pose", "apose"]:
        return "a-pose"
    return text

import spacy
nlp = spacy.load('en_core_web_sm')
# Tokenizer according to https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def get_tokens(caption):
    word_list, pose_list = process_text(caption)
    return ['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))]

def process_tokens(tokens, max_text_len, w_vectorizer):
    if len(tokens) < max_text_len:
        # pad with "unk"
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)

    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    return word_embeddings, pos_one_hots, sent_len, '_'.join(tokens)

def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel-teach/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel


def process_actions(segments: List[List]):
    # segments can be a tuple of size from 1 to N
    if len(segments) == 1:
        return segments
    
    # if there are more than 2 elements
    for i in range(len(segments) - 1): # --> make them consecutive without overlaps
        over = segments[i+1][0] - segments[i][1] # overlap
        segments[i], segments[i+1] = (segments[i][0], segments[i][1] + over//2), (segments[i][1] + over//2 + 1, segments[i+1][1])

    # check all are consecutive
    for i in range(len(segments) - 1):
        assert segments[i][1] + 1 == segments[i+1][0], f'{segments[i][1] + 1} != {segments[i+1][0]}'
    return segments


def timeline_overlaps(arr1: Tuple, arr2: List[Tuple]) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    '''
    Returns the intervals for which:
    (1) arr1 has overlap with
    (2) arr1 is a subset of
    (3) arr1 is a superset of
    '''
    l = arr1[0]
    r = arr1[1]
    inter_sub = []
    inter_super = []
    inter_before = []
    inter_after = []
    for s in arr2:

        if (s[0] > l and s[0] > r) or (s[1] < l and s[1] < r):
            continue
        if s[0] <= l and s[1] >= r:
            inter_sub.append(s)
        if s[0] >= l and s[1] <= r:
            inter_super.append(s)
        if s[0] < l and s[1] < r and s[1] >= l:
            inter_before.append(s)
        if s[0] > l and s[0] <= r and s[1] > r:
            inter_after.append(s)

    return inter_before, inter_after


def segments_sorted(segs_fr: List[List], acts: List) -> Tuple[List[List], List]:
    assert len(segs_fr) == len(acts)
    if len(segs_fr) == 1: return segs_fr, acts
    L = [(segs_fr[i], i) for i in range(len(segs_fr))]
    L.sort()
    sorted_segs_fr, permutation = zip(*L)
    sort_acts = [acts[i] for i in permutation]

    return list(sorted_segs_fr), sort_acts

def extract_single_segments(babel_labels, fps, seqlen):
    seg_ids = []
    seg_acts = []
    is_valid = True
    if babel_labels['frame_ann'] is None:
        is_valid = False

    if is_valid:
        if babel_labels['frame_ann'] is None: # single sequence/action

            # 'transl' 'pose''betas'
            action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
            seg_ids.append([0, seqlen])
            seg_acts.append(fix_spell(action_label))
        else:
            # Get segments
            for seg_an in babel_labels['frame_ann']['labels']:
                action_label = fix_spell(seg_an['proc_label'])

                st_f = int(seg_an['start_t'] * fps) # starting frame
                end_f = min(seqlen, int(seg_an['end_t'] * fps)) # ending frame
                seg_ids.append((st_f, end_f))
                seg_acts.append(action_label)
            # Process segments
            assert len(seg_ids) == len(seg_acts)
    return seg_ids, seg_acts, is_valid


def extract_frame_labels(babel_labels, fps, seqlen, maxframes=200):
    seg_ids = []
    seg_acts = []
    is_valid = True
    babel_key = babel_labels['babel_sid']
    if babel_labels['frame_ann'] is None:
        is_valid = False

    if is_valid:
        if babel_labels['frame_ann'] is None: # single sequence/action

            # 'transl' 'pose''betas'
            action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
            seg_ids.append([0, seqlen])
            seg_acts.append(fix_spell(action_label))
        else:
            # Get segments
            for seg_an in babel_labels['frame_ann']['labels']:
                action_label = fix_spell(seg_an['proc_label'])

                st_f = int(seg_an['start_t'] * fps) # starting frame
                end_f = min(seqlen, int(seg_an['end_t'] * fps)) # ending frame
                seg_ids.append((st_f, end_f))
                seg_acts.append(action_label)
            # Process segments
            assert len(seg_ids) == len(seg_acts)
            import itertools

            seg_ids, seg_acts = segments_sorted(seg_ids, seg_acts)

            # remove excluded actions, if any
            seg_acts_for_pairs = [a for a in seg_acts if a not in EXCLUDED_ACTIONS_WO_TR]
            idx_to_keep = [i for i, a in enumerate(seg_acts) if a not in EXCLUDED_ACTIONS_WO_TR]
            seg_ids_for_pairs = [s for i, s in enumerate(seg_ids) if i in idx_to_keep]
            assert len(seg_acts_for_pairs) == len(seg_ids_for_pairs)

            seg2act = dict(zip(seg_ids_for_pairs, seg_acts_for_pairs))
            seg_ids_for_pairs = list(seg2act.keys()) # remove a few duplicated labels
            # plot_timeline(seg_ids, seg_acts, babel_key)

            overlaps_for_each_seg = {}
            for idx, segment in enumerate(seg_ids_for_pairs):
                # remove the segment of interest
                seg_ids_wo_seg = [x for x in seg_ids_for_pairs if x != segment]
                # calculate the before and after overlaps for the segment of interest
                ov_bef, ov_aft = timeline_overlaps(segment, seg_ids_wo_seg)

                overlaps_for_each_seg[segment] = {}
                overlaps_for_each_seg[segment]['before'] = ov_bef
                overlaps_for_each_seg[segment]['after'] = ov_aft

            # check if relation is symmetric
            for seg_, ov_seg in overlaps_for_each_seg.items():
                for seg2 in ov_seg['before']:
                    assert seg_ in overlaps_for_each_seg[seg2]['after'], f'{seg_} not in {seg2} after. Relationship must be symmetric'
            # if relationship is SYMMETRIC, then we don't need to account for the "before" case, and simply construct the pairs based on the "after" case
            # this is an optimization w.r.t TEACH and PriorMDM
            # REASON: I will reach all the segments, and any segment in "before" is in some segment's "after", so I will reach all the segments/combinations anyway

            # now remove transitions
            # WE BUILD THE COMBINATIONS OF ACTIONS --> [act1, transition, act2, transition, act3, act4, etc], as many as fit inside the "maxframes"
            pairs_s = []
            pairs_a = []
            from copy import deepcopy
            def recursive(ov_per_seg, cur_seg, current_pair=[]):
                ov_seg = ov_per_seg[cur_seg] # before and after segments for the current segment

                # if it is a transition (if not at the beginning), and there are before and after segments (--> we already have the pair)
                if seg2act[cur_seg] == 'transition' and not cur_seg[0] == 0 and ov_seg['before'] and ov_seg['after']: 
                    cur_seg_pairs = list(itertools.product(ov_seg['before'], ov_seg['after'])) # finish pair because it is a transition, so (before, next) are a valid pair.

                else: # if not transition, or if it is the first transition, or if there are no before/after segments
                    after_noT = [x for x in ov_seg['after'] if seg2act[x] != 'transition']
                    # if there are after segments (!= transition, otherwise we will add it next)
                    cur_seg_pairs_af = list(itertools.product([cur_seg], after_noT)) if after_noT else []

                    if after_noT: # if there are after segments, then we can add the current segment to the pair
                        cur_seg_pairs = cur_seg_pairs_af
                    elif current_pair != [] and ov_seg['after']: # skip this segment, and look at the next one (transition)
                        cur_seg_pairs = []
                        for seg in ov_seg['after']:
                            cur_seg_pairs += recursive(ov_per_seg, seg, current_pair)
                        return cur_seg_pairs
                    else: # no more after segments, so we are done
                        return [current_pair, ] if current_pair != [] else [] # END OF RECURSION
                    
                cur_seg_pairs = [current_pair + [b, ] for a, b in cur_seg_pairs] if current_pair else cur_seg_pairs
                # just to be sure
                cur_seg_pairs = [sorted(p, key=lambda item: item[0]) for p in cur_seg_pairs]
                    
                # look at the future until filling the maxframes
                all_seg_pairs = []
                for segs in cur_seg_pairs:
                    # if total length exceeds maxframes, return None
                    if segs[-1][1] - segs[0][0] > maxframes:
                        all_seg_pairs.append(segs[:-1]) # do not look further for this segment, already OVERFLOWS
                    else: # find next pair
                        all_seg_pairs += recursive(ov_per_seg, segs[-1], segs) # look for the next pair, starting from the last segment of the current pair
                    
                return all_seg_pairs

            for seg_, ov_seg in overlaps_for_each_seg.items():
                cur_seg_pairs = recursive(overlaps_for_each_seg, seg_)
                # max of 10 segments per combination, trim if longer
                cur_seg_pairs = [p if len(p) < 10 else p[:10] for p in cur_seg_pairs]

                if len(cur_seg_pairs) > 0:
                    cur_act_pairs = [[seg2act[s] for s in segs] for segs in cur_seg_pairs]
                    pairs_a += cur_act_pairs
                    pairs_s += cur_seg_pairs

            # remove duplicates --> needed because the process above can create duplicates for consecutive non-transition actions
            from more_itertools import unique_everseen
            tmp = zip(pairs_s, pairs_a)
            uniq_tmp = unique_everseen(tmp, key=itemgetter(0))
            segment_pairs = []
            action_pairs = []
            for seg, a in list(uniq_tmp):
                segment_pairs.append(seg)
                action_pairs.append(a)
            if len(segment_pairs) != len(pairs_s):
                pass
                #print(f'{segment_pairs}, {pairs_s}, {len(segment_pairs)} != {len(pairs_s)}')

            assert len(segment_pairs) == len(action_pairs)
            if segment_pairs:
                is_valid = True
                return segment_pairs, action_pairs, is_valid
            else:
                is_valid = False
                return segment_pairs, action_pairs, is_valid
    return seg_ids, seg_acts, is_valid



def load_and_freeze_clip(clip_version):
    import clip
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(clip_model, raw_text, device):
    import clip
    # raw_text - list (batch_size length) of strings with input text prompts
    max_text_len = None 
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2 # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float()


class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self, datapath: str,
                 transforms: Transform,
                 split: str = "train",
                 sampler=None,
                 progress_bar: bool = True,
                 downsample=True,
                 parse_tokens: bool = False,
                 **kwargs):

        self.split = split
        self.parse_tokens = parse_tokens
        self.downsample = downsample
        self.transforms = transforms
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        assert sampler is not None, 'Must inject sampler via constructor'
        self.sampler = sampler
        super().__init__()
        data_for_split = get_split(path=datapath, split=split)
        self.babel_annots = read_json(Path(datapath) / f'../babel-teach/{split}.json')

        motion_data = {}
        texts_data = {}
        durations = {}

        num_bad_actions = 0
        num_bad_short = 0
        valid_data_len = 0
        invalid = 0
        all_data_len = 0
        num_bad_bml = 0
        num_not_kit = 0

        self.precomputed_folder = "./dataset/babel/tmp/custom/"
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{split}_motion_data.pkl')):
            if progress_bar:
                enumerator = enumerate(tqdm(data_for_split, f"Loading BABEL {split}"))
            else:
                enumerator = enumerate(data_for_split)


            for i, sample in enumerator:

                #if i == 300:
                #    break

                all_data_len += 1
                nframes_total = len(sample['poses'])
                last_framerate = sample['fps']
                babel_id = sample['babel_id']
                seg_ids, seg_acts, valid = extract_frame_labels(self.babel_annots[babel_id],
                                                                fps=last_framerate,
                                                                seqlen=nframes_total,
                                                                maxframes=200)

                if not valid:
                    invalid += 1
                    continue

                for index, seg in enumerate(seg_ids):
                    fsegs = process_actions(seg) # make them consecutive without overlaps.
                    frames = np.arange(fsegs[0][0], fsegs[-1][1])
                    duration = [(e-s+1) for s, e in fsegs]
                    duration[-1] -= 1

                    smpl_data = {"poses": 
                                    torch.from_numpy(sample['poses'][frames]).float(), # --> [seq_len, 156] from SMPL-H pose params
                                "trans": 
                                    torch.from_numpy(sample['trans'][frames]).float()} # --> [seq_len, 3]

                    total_duration = np.sum(duration)
                    if not self.sampler.accept(total_duration):
                        num_bad_short += 1
                        continue

                    valid_data_len += 1
                    if len(seg_acts[index]) == 1 and seg_acts[index][0] in EXCLUDED_ACTIONS: # individual excluded sequences
                        num_bad_actions += 1
                        continue

                    from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True) # .rots --> [seq_len, 22, 3, 3], trans --> [seq_len, 3]
                    features = self.transforms.rots2rfeats(smpl_data) # --> [seq_len, 135]
                        
                    texts_data[f'{babel_id}-{index}'] = seg_acts[index]
                    durations[f'{babel_id}-{index}'] = duration
                    motion_data[f'{babel_id}-{index}'] = features
                
            
            if split != "test":
                total = valid_data_len
                logger.info(f"Processed {all_data_len} sequences and found {invalid} invalid cases based on the datatype.")
                percentage = 100 * (num_bad_actions + num_bad_short) / (total + num_bad_short + num_bad_actions)
                logger.info(f"{percentage:.4}% of the sequences which are rejected by the sampler in total.")
                percentage = 100 * num_bad_actions / (total + num_bad_short + num_bad_actions)
                logger.info(
                    f"{percentage:.4}% of the sequence which are rejected by the sampler, because of the excluded actions.")
                percentage = 100 * num_bad_short / (total + num_bad_short + num_bad_actions)
                logger.info(
                    f"{percentage:.4}% of the sequence which are rejected by the sampler, because they are too short(<{self.sampler.min_len / 30} secs) or too long(>{self.sampler.max_len / 30} secs).")
                logger.info(f"Discard from BML: {num_bad_bml}")
                logger.info(f"Discard not KIT: {num_not_kit}")


            if self.parse_tokens:
                tokens_data = {}
                for k, c_texts in tqdm(texts_data.items()):
                    tokens_data.update({k: [get_tokens(t) for t in c_texts]})
                
                with open(os.path.join(self.precomputed_folder, f'{split}_tokens_data.pkl'), 'wb') as f:
                    pickle.dump(tokens_data, f)
            else:
                tokens_data = None

            # store motion_data, texts_data, durations
            print("Storing precomputed data")
            os.makedirs(self.precomputed_folder, exist_ok=True)
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data.pkl'), 'wb') as f:
                pickle.dump(motion_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data.pkl'), 'wb') as f:
                pickle.dump(texts_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations.pkl'), 'wb') as f:
                pickle.dump(durations, f)

            
            # NEW ================ Precompute xyz features
            if split == 'train':
                print('Precomputing xyz features')
                from data_loaders.amass.transforms.smpl import SlimSMPLTransform
                self.transform = SlimSMPLTransform(batch_size=1, name='SlimSMPLTransform', ename='smplnh', normalization=True)
                motions_xyz = {}
                for keyid, motion in tqdm(motion_data.items()):
                    motion_xyz = self.transform.SlimDatastruct(features=motion.unsqueeze(0)).joints.permute(0, 2, 3, 1)
                    motions_xyz[keyid] = motion_xyz.squeeze()

                with open(os.path.join(self.precomputed_folder, f'{split}_motions_xyz.pkl'), 'wb') as f:
                    pickle.dump(motions_xyz, f)
            else:
                motions_xyz = None

        else:
            print("Loading precomputed data")
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data.pkl'), 'rb') as f:
                motion_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data.pkl'), 'rb') as f:
                texts_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations.pkl'), 'rb') as f:
                durations = pickle.load(f)


            if self.parse_tokens and os.path.exists(os.path.join(self.precomputed_folder, f'{split}_tokens_data.pkl')):
                with open(os.path.join(self.precomputed_folder, f'{split}_tokens_data.pkl'), 'rb') as f:
                    tokens_data = pickle.load(f)
            elif self.parse_tokens:
                tokens_data = {}
                for k, c_texts in tqdm(texts_data.items()):
                    tokens_data.update({k: [get_tokens(t) for t in c_texts]})
                
                with open(os.path.join(self.precomputed_folder, f'{split}_tokens_data.pkl'), 'wb') as f:
                    pickle.dump(tokens_data, f)
            else:
                tokens_data = None
            
            if split == 'train':
                with open(os.path.join(self.precomputed_folder, f'{split}_motions_xyz.pkl'), 'rb') as f:
                    motions_xyz = pickle.load(f)
            else:
                motions_xyz = None


        self.motion_data = motion_data
        self.texts_data = texts_data
        self.tokens_data = tokens_data
        self.motions_xyz = motions_xyz

        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        # breakpoint()
        self.keyids = list(self.motion_data.keys())

        self.nfeats = 135

        if split == 'train':
            self.precompute_text_embeddings('ViT-B/32')


    def precompute_text_embeddings(self, clip_version):
        self.clip_embeddings = {}
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{self.split}_clip_embeddings.pkl')):
            print("Precomputing CLIP embeddings")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            clip_model = load_and_freeze_clip(clip_version).to(device)
            for key, texts in tqdm(self.texts_data.items()):
                self.text_embedding = encode_text(clip_model, texts, device)
                self.clip_embeddings[key] = self.text_embedding.cpu().numpy()
            del clip_model

            with open(os.path.join(self.precomputed_folder, f'{self.split}_clip_embeddings.pkl'), 'wb') as f:
                pickle.dump(self.clip_embeddings, f)
        else:
            print("Loading precomputed CLIP embeddings")
            with open(os.path.join(self.precomputed_folder, f'{self.split}_clip_embeddings.pkl'), 'rb') as f:
                self.clip_embeddings = pickle.load(f)

    def inv_transform(self, motion):
        return motion # no need to inverse transform

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.motion_data[keyid][frame_ix]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def _load_tokens(self, keyid):
        sequences = self.tokens_data[keyid]
        return sequences

    def _load_actions(self, keyid):
        actions_all = self.action_datas[keyid]
        return actions_all

    def load_keyid(self, keyid, mode='train'):
        all_texts = self._load_text(keyid)
        if len(all_texts) == 1:
            text = all_texts[0]
        else:
            text = ""
            for t in all_texts:
                text += t + " -- "

        #if self.parse_tokens:
        #    tokens = self._load_tokens(keyid)

        if mode == 'train':
            features = self.motion_data[keyid]
            length = np.sum(self._num_frames_in_sequence[keyid]) # sum of all the durations

            element = {"features": features,
                        "length": length,
                        #"is_transition": torch.zeros(length), # we don't use this in our work
                        "keyid": keyid,
                        "text": text,
                        "features_xyz": self.motions_xyz[keyid] if self.split == 'train' else torch.zeros(1),
                        # metadata for multi-text conditioning
                        "all_texts": self.texts_data[keyid],
                        "all_lengths": self._num_frames_in_sequence[keyid],
                        }
            if self.split == 'train':
                element.update({"text_embeddings": self.clip_embeddings[keyid]})
        else:
            raise ValueError("mdm project - you should never use mode other than train in our scope")
        return element


    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"




class BABEL_SingleEval(BABEL):

    def __init__(self, datapath: str,
                 transforms: Transform,
                 split: str = "train",
                 sampler=None,
                 progress_bar: bool = True,
                 w_vectorizer=None,
                 opt=None,
                 **kwargs):

        self.w_vectorizer = w_vectorizer
        self.opt = opt

        self.split = split
        self.crop_sample = kwargs.get('cropping_sampler', False)

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = sampler.max_len

        self.transforms = transforms
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        assert sampler is not None, 'Must inject sampler via constructor'
        self.sampler = sampler

        data_for_split = get_split(path=datapath, split=split)
        self.babel_annots = read_json(Path(datapath) / f'../babel-teach/{split}.json')

        motion_data = {}
        texts_data = {}
        durations = {}

        excluded_too_short = 0
        excluded_too_long = 0
        excluded_action = 0
        cropped = 0


        self.precomputed_folder = "./dataset/babel/tmp/custom/"
        suffix = f"_single_eval_{self.sampler.min_len}_{self.sampler.max_len}_{self.crop_sample}"
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl')):
            if progress_bar:
                enumerator = enumerate(tqdm(data_for_split, f"Loading BABEL {split}"))
            else:
                enumerator = enumerate(data_for_split)

            for i, sample in enumerator:
                nframes_total = len(sample['poses'])
                last_framerate = sample['fps']
                babel_id = sample['babel_id']
                seg_ids, seg_acts, valid = extract_single_segments(self.babel_annots[babel_id],
                                                                fps=last_framerate,
                                                                seqlen=nframes_total
                                                                )

                if not valid:
                    continue

                for index, seg in enumerate(seg_ids):
                    duration = seg[1] - seg[0]
                    frames = np.arange(seg[0], seg[1])

                    accept = self.sampler.accept(duration)

                    if seg_acts[index] in EXCLUDED_ACTIONS: # exclude transitions here
                        excluded_action += 1
                        continue

                    if not accept and (not self.crop_sample or not self.sampler.can_be_cropped(duration)):
                        if duration < self.sampler.min_len:
                            excluded_too_short += 1
                        elif duration > self.sampler.max_len:
                            excluded_too_long += 1
                        continue
                    elif not accept: # --> crop
                        cropped += 1
                        t0, t1 = self.sampler.sample(duration)
                        seg = [seg[0] + t0, seg[0] + t1]
                        duration = seg[1] - seg[0] - 1

                    smpl_data = {"poses": 
                                    torch.from_numpy(sample['poses'][frames]).float(), # --> [seq_len, 156] from SMPL-H pose params
                                "trans": 
                                    torch.from_numpy(sample['trans'][frames]).float()} # --> [seq_len, 3]

                    from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True) # .rots --> [seq_len, 22, 3, 3], trans --> [seq_len, 3]
                    features = self.transforms.rots2rfeats(smpl_data) # --> [seq_len, 135]
                        
                    texts_data[f'{babel_id}-{index}'] = standarize_text(seg_acts[index])
                    durations[f'{babel_id}-{index}'] = duration
                    motion_data[f'{babel_id}-{index}'] = features
            
        

            # store motion_data, texts_data, durations
            print("Storing precomputed data")
            os.makedirs(self.precomputed_folder, exist_ok=True)
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl'), 'wb') as f:
                pickle.dump(motion_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data{suffix}.pkl'), 'wb') as f:
                pickle.dump(texts_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations{suffix}.pkl'), 'wb') as f:
                pickle.dump(durations, f)


        else:
            print("Loading precomputed data")
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl'), 'rb') as f:
                motion_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data{suffix}.pkl'), 'rb') as f:
                texts_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations{suffix}.pkl'), 'rb') as f:
                durations = pickle.load(f)

        #print(f'Excluded {excluded_action} sequences because of action')
        #print(f'Excluded {excluded_too_short} sequences because too short')
        #print(f'Excluded {excluded_too_long} sequences because too long')
        #print(f'Cropped {cropped} sequences')
        #print(f'Total {len(motion_data)} sequences')

        self.motion_data = motion_data
        self.texts_data = texts_data

        # if not tiny:
        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        # breakpoint()
        self.keyids = list(self.motion_data.keys())

        self.nfeats = 135


    def load_keyid(self, keyid, mode='train'):
        text = self._load_text(keyid)
        
        features = self.motion_data[keyid]
        length = np.sum(self._num_frames_in_sequence[keyid]) # sum of all the durations

        element = {"features": features,
                    "length": length,
                    #"is_transition": torch.zeros(length), # we don't use this in our work
                    "tokens": get_tokens(text),
                    "keyid": keyid,
                    "text": text,
                    }
        return element
    
    def process_tokens(self, tokens):
        return process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

    def __getitem__(self, item):
        keyid = self._split_index[item]
        batch = self.load_keyid(keyid, mode='train')

        # Randomly choose a motion from batch
        caption = batch['text']
        tokens = batch['tokens']
        motion = batch['features']
        m_length = batch['length']

        word_embeddings, pos_one_hots, sent_len, tokens = process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

        idx = random.randint(0, abs(len(motion) - m_length))
        motion = motion[idx:idx+m_length]

        if m_length <= self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
                                     
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens


class BABEL_TransitionsEval(BABEL_SingleEval):

    def __init__(self, datapath: str,
                 transforms: Transform,
                 split: str = "train",
                 sampler=None,
                 progress_bar: bool = True,
                 w_vectorizer=None,
                 opt=None,
                 **kwargs):

        self.w_vectorizer = w_vectorizer
        self.opt = opt

        self.split = split
        self.crop_sample = kwargs.get('cropping_sampler', False)

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = sampler.max_len

        self.transforms = transforms
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        assert sampler is not None, 'Must inject sampler via constructor'
        self.sampler = sampler

        data_for_split = get_split(path=datapath, split=split)
        self.babel_annots = read_json(Path(datapath) / f'../babel-teach/{split}.json')

        motion_data = {}
        texts_data = {}
        durations = {}

        self.precomputed_folder = "./dataset/babel/tmp/custom/"
        suffix = f"_trans_eval_{self.sampler.min_len}_{self.sampler.max_len}_{self.crop_sample}"
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl')):
            if progress_bar:
                enumerator = enumerate(tqdm(data_for_split, f"Loading BABEL {split}"))
            else:
                enumerator = enumerate(data_for_split)

            for i, sample in enumerator:
                nframes_total = len(sample['poses'])
                babel_id = sample['babel_id']
                
                # cut the sequence into segments of random size between self.sampler.min_len and self.sampler.max_len
                avg = (self.sampler.min_len + self.sampler.max_len) // 2
                num_samples = nframes_total // avg # --> num of random subsequences
                for j in range(num_samples):
                    duration = np.random.randint(self.sampler.min_len, self.sampler.max_len) if self.sampler.max_len > self.sampler.min_len else self.sampler.min_len
                    t0 = np.random.randint(0, nframes_total - duration) if nframes_total > duration else 0
                    t1 = t0 + duration

                    smpl_data = {"poses": 
                                    torch.from_numpy(sample['poses'][t0:t1]).float(), # --> [seq_len, 156] from SMPL-H pose params
                                "trans": 
                                    torch.from_numpy(sample['trans'][t0:t1]).float()} # --> [seq_len, 3]

                    from data_loaders.amass.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True) # .rots --> [seq_len, 22, 3, 3], trans --> [seq_len, 3]
                    features = self.transforms.rots2rfeats(smpl_data) # --> [seq_len, 135]
                        
                    texts_data[f'{babel_id}_{j}'] = ""
                    durations[f'{babel_id}_{j}'] = duration
                    motion_data[f'{babel_id}_{j}'] = features
            
            # store motion_data, texts_data, durations
            print("Storing precomputed data")
            os.makedirs(self.precomputed_folder, exist_ok=True)
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl'), 'wb') as f:
                pickle.dump(motion_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data{suffix}.pkl'), 'wb') as f:
                pickle.dump(texts_data, f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations{suffix}.pkl'), 'wb') as f:
                pickle.dump(durations, f)


        else:
            print("Loading precomputed data")
            with open(os.path.join(self.precomputed_folder, f'{split}_motion_data{suffix}.pkl'), 'rb') as f:
                motion_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_texts_data{suffix}.pkl'), 'rb') as f:
                texts_data = pickle.load(f)
            with open(os.path.join(self.precomputed_folder, f'{split}_durations{suffix}.pkl'), 'rb') as f:
                durations = pickle.load(f)

        self.motion_data = motion_data
        self.texts_data = texts_data

        # if not tiny:
        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        # breakpoint()
        self.keyids = list(self.motion_data.keys())

        self.nfeats = 135
