from torch.utils.data import DataLoader
from data_loaders.datasets_composition import CompMDMUnfoldingGeneratedDataset
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


# our loader
def get_mdm_loader(args, model, diffusion, batch_size, eval_file, dataset, max_motion_length=200,
                   precomputed_folder=None, scenario=""):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    # all batch computed as single segment, all actions in a single sequence
    dataset = CompMDMUnfoldingGeneratedDataset(args, model, diffusion, max_motion_length, eval_file, w_vectorizer=dataset.w_vectorizer, opt=dataset.opt,
                                            precomputed_folder=precomputed_folder, scenario=scenario)
    mm_motion_loader = None

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    motion_loaders = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=0, shuffle=False)

    print('Generated Dataset Loading Completed!!!')

    return motion_loaders, mm_motion_loader