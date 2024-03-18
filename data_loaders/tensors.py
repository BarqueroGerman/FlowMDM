import torch
import numpy as np

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def babel_eval_collate(batch):
    try:
        adapted_batch = [{
            'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'is_transition': torch.from_numpy(b[7]),
        } for b in batch]
    except TypeError:
        print(5)
    return collate(adapted_batch)

def babel_collate(batch):
    from data_loaders.amass.tools import collate_pairs_and_text
    batch = collate_pairs_and_text(batch)
    bs = len(batch['motion_feats'])
    adapted_batch = []
    for ii in range(bs):
        adapted_batch.append({
            'inp': batch['motion_feats'][ii].permute(1, 0).unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': batch['text'][ii],
            'lengths': batch['length'][ii],
            #'is_transition': batch['is_transition'][ii],
            })
        if 'motion_feats_xyz' in batch:
            adapted_batch[-1]['inp_xyz'] = batch['motion_feats_xyz'][ii]

        # metadata for multi-text conditioning
        if 'all_lengths' in batch and 'all_texts' in batch:
            adapted_batch[-1]['all_lengths'] = batch['all_lengths'][ii]
            adapted_batch[-1]['all_texts'] = batch['all_texts'][ii]
        if 'text_embeddings' in batch:
            adapted_batch[-1]['text_embeddings'] = batch['text_embeddings'][ii]
    return collate(adapted_batch)

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    lenbatchTensor = torch.as_tensor(lenbatch)
    max_seq_len = max(lenbatchTensor)
    databatch = [b['inp'][..., :max_seq_len] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, max_seq_len).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'inp_xyz' in notnone_batches[0]:
        databatchTensor = collate_tensors([b['inp_xyz'] for b in notnone_batches])
        cond['y'].update({'inp_xyz': databatchTensor})

    if 'all_texts' in notnone_batches[0] and 'all_lengths' in notnone_batches[0]:
        all_texts = [b['all_texts'] for b in notnone_batches]
        cond['y'].update({'all_texts': all_texts})
        all_lengths = [b['all_lengths'] for b in notnone_batches]
        cond['y'].update({'all_lengths': all_lengths})

        num_max_segs = max([len(x) for x in all_texts])

        device = databatchTensor.device
        batch_size = databatchTensor.shape[0]
        # 'conditions_mask' has shape [I, T, N] where I is the number of different conditions, N is batch size, T is sequence length.
        conditions_mask = torch.zeros((num_max_segs, max_seq_len, batch_size), device=device, dtype=torch.bool)
        for i, c_all_lengths in enumerate(all_lengths):
            s = 0
            for j, length in enumerate(c_all_lengths):
                conditions_mask[j, s:s+length, i] = True # all batch elements have the same instructions
                s += length
        cond['y'].update({'conditions_mask': conditions_mask})

        pos_pe_abs = torch.zeros((batch_size, max_seq_len), device=device, dtype=torch.float32)
        pe_bias = torch.full((batch_size, max_seq_len, max_seq_len), float('-inf'), device=device, dtype=torch.float32)
        for i, c_all_lengths in enumerate(all_lengths):
            s = 0 # start
            for length in c_all_lengths:
                pos_pe_abs[i, s:s+length] = torch.arange(length, device=device, dtype=torch.float32)
                pe_bias[i, s:s+length, s:s+length] = 0 # only attend to the segment for the absolute modeling part of the schedule
                s += length

        cond['y']['pe_bias'] = pe_bias # in MDM forward, it is selected according to the mixed schedule if active
        cond['y']['pos_pe_abs'] = pos_pe_abs

        if 'text_embeddings' in notnone_batches[0]:
            # prepare the precomputed text embeddings
            texts_list = cond['y']['all_texts']
            # make sure all texts have the same length. If not, add dummy tokens
            max_len = max([len(texts) for texts in texts_list])
            text_embeddings = torch.zeros((max_len, batch_size, 512), device=device, dtype=torch.float32)
            for i, texts in enumerate(texts_list):
                text_embeddings[:len(texts), i, :] = notnone_batches[i]["text_embeddings"][:len(texts)]
            cond['y']['text_embeddings'] = text_embeddings
            
    elif 'text_embeddings' in notnone_batches[0]:
        text_embeddings = [b['text_embeddings'] for b in notnone_batches]
        text_embeddings = collate_tensors(text_embeddings)
        cond['y'].update({'text_embeddings': text_embeddings.to(databatchTensor.device)})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].transpose((1, 0))).float().unsqueeze(1) if len(b[4].shape) == 2 else torch.tensor(b[4].transpose((1, 2, 0))).float(), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5]
    } for b in batch]
    if len(batch[0]) >= 9 and not isinstance(batch[0][8], list): # then load precomputed text embeddings
        for i, b in enumerate(adapted_batch):
            b['text_embeddings'] = torch.tensor(batch[i][8]).float()
    return collate(adapted_batch)
