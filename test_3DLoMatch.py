import json
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict as edict
from libs.loss import *
from datasets.ThreeDMatch import ThreeDLOMatchTest
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from evaluation.benchmark_utils_predator import *
from utils.timer import Timer
from utils.SE3 import *
import os
set_seed()


def eval_3DMatch_scene(model, scene_ind, dloader, config, args):
    correct_num = 0
    correct_ratio = 0
    seed_precision = 0
    seed_num = 0
    num_pair = dloader.dataset.__len__()
    stats = np.zeros([num_pair, 13])
    final_poses = np.zeros([num_pair, 4, 4])
    dloader_iter = dloader.__iter__()
    
    class_loss = ClassificationLoss()
    evaluate_metric = TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre)
    data_timer, model_timer = Timer(), Timer()

    with torch.no_grad():
        for i in tqdm(range(num_pair), ncols=100):
            #################################
            # load data 
            #################################
            data_timer.tic()
            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels, scene, src_id, tgt_id = next(dloader_iter)
            corr, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels = \
                corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), src_normal.cuda(), tgt_normal.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'src_normal': src_normal,
                'tgt_normal': tgt_normal,
                'labels': gt_labels,
                'testing': True,
            }
            data_time = data_timer.toc()

            #################################
            # forward pass 
            #################################
            model_timer.tic()
            res = model(data)
            model_time = model_timer.toc()
            
            pred_trans = res['final_trans']
            logits = res['logits']
            confidence = res['confidence']

            # Use logits directly
            pred_labels = (logits > 0).int()

            # Generate Hypotheses explicitly using already computed features/confidence
            # Note: Modified to accept pre-computed results to save memory
            sampled_trans = model.predict_hypotheses(data, num_seeds=100, k=20, res=res).squeeze(0)

            # Evaluate Hypotheses quality
            if sampled_trans.dim() == 3 and sampled_trans.shape[0] > 0:
                src_expanded = src_keypts.repeat(sampled_trans.shape[0], 1, 1)
                R_hyp = sampled_trans[:, :3, :3]
                t_hyp = sampled_trans[:, :3, 3].unsqueeze(1)

                src_transformed = torch.matmul(src_expanded, R_hyp.transpose(1, 2)) + t_hyp
                tgt_expanded = tgt_keypts.repeat(sampled_trans.shape[0], 1, 1)
                dists = torch.norm(src_transformed - tgt_expanded, dim=-1)
                inlier_counts = (dists < config.inlier_threshold).sum(dim=1)

                best_idx = torch.argmax(inlier_counts)
                best_trans = sampled_trans[best_idx].unsqueeze(0)

                pred_trans = best_trans
                
                # Update pred_labels based on the best hypothesis for consistency
                warped_src = transform(src_keypts, pred_trans)
                final_dists = torch.norm(warped_src - tgt_keypts, dim=-1)
                pred_labels = (final_dists < config.inlier_threshold).float()

            if args.use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, src_normal, tgt_normal, pred_trans, config.inlier_threshold)

            class_stats = class_loss(pred_labels, gt_labels)
            recall_val, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            # Update summary metrics
            seed_precision += gt_labels[0, res['seeds'][0]].sum().item() / res['seeds'].size(1) * 100.0
            seed_num += gt_labels[0, res['seeds'][0]].sum().item()
            if recall_val > 0:
                correct_num += 1

            # Save statistics
            stats[i, 0] = float(recall_val / 100.0)
            stats[i, 1] = float(Re)
            stats[i, 2] = float(Te)
            stats[i, 3] = int(torch.sum(gt_labels))
            stats[i, 4] = float(torch.mean(gt_labels.float()))
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0]))
            stats[i, 6] = float(class_stats['precision'])
            stats[i, 7] = float(class_stats['recall'])
            stats[i, 8] = float(class_stats['f1'])
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = int(src_id)
            stats[i, 12] = int(tgt_id)
            final_poses[i] = pred_trans[0].detach().cpu().numpy()
            
            torch.cuda.empty_cache()

    return stats, final_poses, seed_num, seed_precision, correct_num, correct_ratio


def eval_3DMatch(model, config, args):
    dset = ThreeDLOMatchTest(root=config.root,
                            descriptor=args.descriptor,
                            inlier_threshold=config.inlier_threshold,
                            num_node=args.num_points,
                            augment_axis=0,
                            augment_rotation=0.00,
                            augment_translation=0.0,
                            )
    dloader = get_dataloader(dset, batch_size=1, num_workers=0, shuffle=False)
    os.makedirs('logs/3dlomatch', exist_ok=True)
    if os.path.isfile('logs/3dlomatch/'+args.descriptor+'.txt'):
        os.remove('logs/3dlomatch/'+args.descriptor+'.txt')
        
    allpair_stats, allpair_poses, avg_seed_num, avg_seed_precision, avg_correct_num, avg_correct_ratio = eval_3DMatch_scene(model, 0, dloader, config, args)

    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    
    if args.descriptor == 'predator':
        benchmark_predator(allpair_poses, gt_folder='benchmarks/3DLoMatch')

    return allpair_stats


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='TransformerPruner_3DMatch', type=str, help='snapshot dir')
    parser.add_argument('--descriptor', default='fpfh', type=str)
    parser.add_argument('--num_points', default='5000', type=str)
    parser.add_argument('--use_icp', default=False, type=str2bool)
    args = parser.parse_args()
    
    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    config.inlier_threshold = 0.1
    config.re_thre = 15
    config.te_thre = 30
    
    log_filename = f'logs/3DLoMatch_{args.chosen_snapshot}-{args.descriptor}-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   

    config.mode = "test"
    from models.outlier_pruner import MethodName
    model = MethodName(config)

    miss = model.load_state_dict(torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'), strict=True)
    print(miss)
    model.eval()

    stats = eval_3DMatch(model.cuda(), config, args)