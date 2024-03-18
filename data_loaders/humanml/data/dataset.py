from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import pickle

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
import torch

from data_loaders.amass.babel_flowmdm import BABEL_SingleEval, BABEL_TransitionsEval


MOTION_TYPES = [
    '_0',
    '_1',
    '_0_with_transition',
    '_1_with_transition',
]

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


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


'''For use of training text motion matching model, and evaluations'''
class HumanML3D_Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, **kwargs):
        self.opt = opt
        self.mean = mean
        self.std = std
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0

        # start loading dataset
        self.num_frames = num_frames if num_frames else False
        self.max_motion_length = opt.max_motion_length
        if (self.num_frames == False) or type(self.num_frames)==int:
            self.min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        else:
            self.min_motion_len = self.num_frames[0]
            self.max_motion_length = self.num_frames[1]

        self.precomputed_folder = "./dataset/HumanML3D/tmp/"
        os.makedirs(self.precomputed_folder, exist_ok=True)
        suffix = f"{num_frames}" if num_frames == False or type(num_frames)==int else f"{num_frames[0]}_{num_frames[1]}"
        self.split = split_file.split('/')[-1].split('.')[0]
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl')):
            from data_loaders.amass.babel_flowmdm import load_and_freeze_clip, encode_text
            clip_model = load_and_freeze_clip('ViT-B/32').to('cuda')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            data_dict = {}
            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
            id_list = id_list[:size]

            new_name_list = []
            length_list = []
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if self.split == 'train':
                                text_dict['text_embedding'] = encode_text(clip_model, [caption], device).cpu().numpy()
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    if self.num_frames != False:
                                        if len(n_motion) >= self.max_motion_length:
                                            bias = random.randint(0, len(n_motion) - self.max_motion_length)
                                            data_dict[new_name] = {'motion': n_motion[bias: bias+self.max_motion_length],
                                                                'length': self.max_motion_length,
                                                                'text': [text_dict]}
                                            length_list.append(self.max_motion_length)

                                        else:
                                            data_dict[new_name] = {'motion': n_motion,
                                                                'length': len(n_motion),
                                                                'text': [text_dict]}
                                            length_list.append(len(n_motion))

                                    else:
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        length_list.append(len(n_motion))

                                    new_name_list.append(new_name)
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        if self.num_frames != False:
                            if len(motion) >= self.max_motion_length:
                                bias = random.randint(0, len(motion) - self.max_motion_length)
                                data_dict[name] = {'motion': motion[bias: bias + self.max_motion_length],
                                                    'length': self.max_motion_length,
                                                    'text': [text_dict]}
                                length_list.append(self.max_motion_length)

                            else:
                                data_dict[name] = {'motion': motion,
                                                'length': len(motion),
                                                'text': text_data}
                                length_list.append(len(motion))

                        else:
                            data_dict[name] = {'motion': motion,
                                            'length': len(motion),
                                            'text': text_data}
                            length_list.append(len(motion))

                        new_name_list.append(name)
                except Exception as e:
                    print(e)
                    pass

            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            self.length_arr = np.array(length_list)
            self.data_dict = data_dict
            self.name_list = name_list

            data_to_store = {'data_dict': data_dict, 'name_list': name_list, 'length_list': length_list}
            with open(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl'), 'wb') as f:
                pickle.dump(data_to_store, f)
        else:
            with open(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl'), 'rb') as f:
                data_to_store = pickle.load(f)
                self.data_dict = data_to_store['data_dict']     
                self.name_list = data_to_store['name_list']
                self.length_arr = np.array(data_to_store['length_list'])


                #data_to_store = {'data_dict': data_dict, 'name_list': name_list, 'length_list': length_list}
                #with open(os.path.join(self.precomputed_folder, f'{split}_data_{suffix}.pkl'), 'wb') as f:
                #    pickle.dump(data_to_store, f)
        
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer
    
    def process_tokens(self, tokens):
        return process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

    def __getitem__(self, item):
        #print(self.pointer, item)
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        text_embedding = text_data['text_embedding'].squeeze() if self.split == "train" else []

        word_embeddings, pos_one_hots, sent_len, tokens = self.process_tokens(tokens)

        m_length = max(m_length, self.min_motion_len)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, [], text_embedding # last [] is for transition sequence, which is not used in this dataset


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, load_mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.load_mode = load_mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'
        self.split = split

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.load_mode = load_mode
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        # self.mean and self.std used by the getter function
        if load_mode == 'gt': # GT is always used to eval GT --> from ORIGINAL UNNORMALIZED to evaluators normalization
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif load_mode in ['train', 'eval', 'gen']: # from ORIGINAL UNNORMALIZED to training normalization
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
        self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')

        self.t2m_dataset = HumanML3D_Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)
        
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class BABEL_eval(data.Dataset):
    def __init__(self, load_mode, datapath, transforms, sampler, opt, split="train", **kwargs):
        self.load_mode = load_mode

        self.split = split
        self.datapath = datapath
        abs_base_path = f'.'

        if opt is None:
            self.opt_path = './dataset/humanml_opt.txt'
            # Configurations of T2M dataset and KIT dataset is almost the same
            dataset_opt_path = pjoin(abs_base_path, self.opt_path)
            device = None  # torch.device('cuda:4') # This param is not in use in this context
            opt = get_opt(dataset_opt_path, device)
            opt.data_root = pjoin('dataset', 'babel')
            opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
            opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
            opt.text_dir = pjoin(abs_base_path, opt.text_dir)
            opt.model_dir = None
            opt.checkpoints_dir = '.'
            opt.data_root = pjoin(abs_base_path, opt.data_root)
            opt.save_root = pjoin(abs_base_path, opt.save_root)
            opt.meta_dir = './dataset'
            opt.dim_pose = 135
            opt.foot_contact_entries = 0
            opt.dataset_name = 'babel'
            opt.decomp_name = 'Decomp_SP001_SM001_H512_babel_2700epoch'
            opt.meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
            opt.min_motion_length = sampler.min_len # must be at least window size
            opt.max_motion_length = sampler.max_len
        self.opt = opt

        print('Loading dataset %s ...' % opt.dataset_name)

        self.dataset_name = opt.dataset_name
        self.dataname = opt.dataset_name
        self.sampler = sampler
        self.transforms = transforms

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        if "transitions" in load_mode:
            self.t2m_dataset = BABEL_TransitionsEval(
                split=self.split,
                datapath=self.datapath,
                transforms=self.transforms,
                opt=self.opt,
                w_vectorizer=self.w_vectorizer, sampler=self.sampler,
                cropping_sampler=kwargs.get('cropping_sampler', False)
            )
        else:
            self.t2m_dataset = BABEL_SingleEval(
                split=self.split,
                datapath=self.datapath,
                transforms=self.transforms,
                opt=self.opt,
                w_vectorizer=self.w_vectorizer, sampler=self.sampler,
                cropping_sampler=kwargs.get('cropping_sampler', False)
            )

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def inv_transform(self, data):
        return data

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
    

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, load_mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(load_mode, datapath, split, **kwargs)

