import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.metrics import calculate_jerk, calculate_top_k, calculate_diversity, calculate_multimodality, euclidean_distance_matrix, calculate_activation_statistics, calculate_frechet_distance
import torch
import os
import matplotlib.pyplot as plt
import json

def evaluate_jerk(motion_loaders):
    jerk_score_dict = {}
    auj_score_dict = {}
    auj_plot_values = {}
    #print('========== Evaluating Jerk ==========')
    for model_name, motion_loader in motion_loaders.items():
        all_jerks = []
        for idx, batch in enumerate(motion_loader):#tqdm(enumerate(motion_loader)):
            motions = batch[4] # [bs, seq_len, nfeats]
            lengths = batch[5] # [bs]
            if motions.shape[-1] == 263: # HUMANML3D ============
                GT_jerk = 0.033031363 # --> extracted from the GT
                #[bs, nfeats, 1, seq_len]
                n_joints = 22 # HumanML --> 22
                motions = motion_loader.dataset.inv_transform(motions) # we need to recover the original denormed values.
                motions = recover_from_ric(motions.float(), n_joints) # --> [bs, seqlen, njoints, 3]
            elif motions.shape[-1] == 135: # BABEL ============
                from data_loaders.amass.transforms import SlimSMPLTransform
                transform = SlimSMPLTransform(batch_size=8, name='SlimSMPLTransform', ename='smplnh', normalization=True)
                GT_jerk = 0.016383045 # --> extracted from the GT
                motions = motions.reshape(-1, motions.shape[-1])
                datastruct = transform.SlimDatastruct(features=motions.float())
                motions = datastruct.joints
                motions = motions.reshape(lengths.shape[0], -1, motions.shape[-2], motions.shape[-1]) # --> [SEQ, 22, 3]
            else:
                raise ValueError(f'Unsupported motion loader [{model_name}]')
            
            batch_jerk = calculate_jerk(motions.cpu().numpy(), lengths.cpu().numpy()) # --> [BS, SEQ]
            all_jerks.append(batch_jerk)

        all_jerks = np.concatenate(all_jerks, axis=0) # --> [BS, SEQ]
        seq_jerks = all_jerks.mean(axis=0) # --> [SEQ] --> mean jerk per frame in the seq

        auj_plot_values[model_name] = seq_jerks
        diff = seq_jerks - GT_jerk
        auj_score_dict[model_name] = np.sum(np.abs(diff)) # Area Under Jerk Curve
        jerk_score_dict[model_name] = seq_jerks.max() # Jerk --> max jerk along the sequence

        print(f'---> [{model_name}] PeakJerk: {jerk_score_dict[model_name]:.4f} AUJ: {auj_score_dict[model_name]:.4f}')

    return jerk_score_dict, auj_score_dict, auj_plot_values


def evaluate_matching_score(eval_wrapper, motion_loaders, file, compute_precision=True, compute_matching=True):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    #print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens = batch[:6]
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                if compute_matching or compute_precision:
                    dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                        motion_embeddings.cpu().numpy())
                if compute_matching:
                    matching_score_sum += dist_mat.trace()

                if compute_precision:
                    unique_idces = np.unique(captions, return_index=True)[1] # remove duplicate captions in the top-k calculation
                    argsmax = np.argsort(dist_mat[unique_idces][:, unique_idces], axis=1)
                    top_k_mat = calculate_top_k(argsmax, top_k=3)
                    corrector = text_embeddings.shape[0] / top_k_mat.shape[0] # correct for the duplicate captions
                    top_k_count += (top_k_mat.sum(axis=0) * corrector)

                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)

            if compute_matching:
                matching_score = matching_score_sum / all_size
                match_score_dict[motion_loader_name] = matching_score
            if compute_precision:
                R_precision = top_k_count / all_size
                R_precision_dict[motion_loader_name] = R_precision

            activation_dict[motion_loader_name] = all_motion_embeddings

        if compute_matching:
            print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
            print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        if compute_precision:
            line = f'---> [{motion_loader_name}] R_precision: '
            for i in range(len(R_precision)):
                line += '(top %d): %.4f ' % (i+1, R_precision[i])
            print(line)
            print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    #print('========== Evaluating FID ==========')
    # compute reference mu and cov
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            motions, m_lens = batch[4:6]
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # compute our mu and cov for FID computation
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    #print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    #print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def generate_plot_PJ(path):
    name_path = path.split("/")[-1].split(".")[0]
    output_dir = os.path.dirname(path)

    # Function x**(1/2)
    def forward(x):
        return x**(1/2)

    def inverse(x):
        return x**2

    title = "Transition jerk"
    results_json = json.load(open(path, 'r'))
    values = results_json["transition"]["PeakJerk"]
    transition_length = len(values)

    # plot figure
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Transition")
    ax.set_ylabel("Maximum jerk over joints")
    ax.set_yscale('function', functions=(forward, inverse))
    ax.set_xlim((0, transition_length-4))
    # xticks must contain (-transition_length//2, middle, +transition_length//2)
    ax.set_xticks([0, transition_length//2-2, transition_length-4])
    ax.set_xticklabels([r'$\tau-\frac{L_{tr}}{2}$', r'$\tau$', r'$\tau+L_{tr}/2$'])
    
    ax.plot(values)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name_path + "_transition_PJ" + ".png"))
    plt.close()