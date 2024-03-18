from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name, load_mode):
    if name == "babel":
        if 'gt' in load_mode or load_mode == 'evaluator_train': # reference motion for evaluation
            from data_loaders.humanml.data.dataset import BABEL_eval
            return BABEL_eval
        elif load_mode == 'gen':
            from data_loaders.amass.babel import BABEL
            return BABEL
        elif load_mode == 'train':
            from data_loaders.amass.babel_flowmdm import BABEL
            return BABEL
        else:
            raise ValueError(f'Unsupported load_mode name [{load_mode}]')
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, load_mode='train'):
    print(name, load_mode)
    if "gt" in load_mode:
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml"]:
        return t2m_collate
    elif name == 'babel' and load_mode != "evaluator_train":
        from data_loaders.tensors import babel_collate
        return babel_collate
    elif name == 'babel':
        from data_loaders.humanml.data.dataset import collate_fn as sorted_collate
        return sorted_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', load_mode='train', **kwargs):
    DATA = get_dataset_class(name, load_mode)
    pose_rep = kwargs.get('pose_rep', 'rot6d')
    if name in ["humanml"]:
        load_mode = "gt" if "gt" in load_mode else load_mode
        dataset = DATA(load_mode, split=split, pose_rep=pose_rep, num_frames=num_frames)
    elif name == "babel":
        cropping_sampler = kwargs.get('cropping_sampler', False)
        opt = kwargs.get('opt', None)
        batch_size = kwargs.get('batch_size', None)
        from data_loaders.amass.transforms import SlimSMPLTransform
        from data_loaders.amass.sampling import FrameSampler
        if ((split=='val') and (cropping_sampler==True)):
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True, canonicalize=False)
        else:
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        sampler = FrameSampler(min_len=num_frames[0], max_len=num_frames[1])
        dataset = DATA(split=split,
                       datapath='./dataset/babel/babel-smplh-30fps-male',
                       transforms=transform, load_mode=load_mode, opt=opt, sampler=sampler,
                       cropping_sampler=cropping_sampler)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, num_workers=8, split='train', load_mode='train', shuffle=True, drop_last=True, **kwargs):
    dataset = get_dataset(name, num_frames, split, load_mode, batch_size=batch_size, **kwargs)
    collate = get_collate_fn(name, load_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    )

    return loader