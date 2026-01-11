import json
import os.path
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
from datasets.ThreeDMatch import ThreeDMatchTest
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from evaluation.benchmark_utils_predator import *
from utils.timer import Timer
set_seed()

def eval_3DMatch_scene(model, scene, scene_ind, dloader, config, args):
    """
    Evaluate our model on 3DMatch testset [scene]
    """
    correct_num = 0
    correct_ratio = 0
    seed_precision = 0
    seed_num = 0
    num_pair = dloader.dataset.__len__()
    
    # Stats columns:
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 13])
    dloader_iter = dloader.__iter__()

    class_loss = ClassificationLoss()
    evaluate_metric = TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre)
    data_timer, model_timer = Timer(), Timer()

    final_poses = np.zeros([num_pair, 4, 4])

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

            # Hypothesis Generation (Seed -> Subset -> SVD)
            res = model(data)
            pred_trans = res['final_trans']
            logits = res['logits']
            confidence = res['confidence']
            pred_labels = (logits > 0).int()

            sampled_trans = model.predict_hypotheses(data, num_seeds=100, k=20).squeeze(0)

            model_time = model_timer.toc()

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
            
            # Optional: RANSAC Refinement
            if args.solver == 'RANSAC':
                src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
                tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
                
                # Filter correspondences using predicted labels
                pred_inliers = np.where(pred_labels[0].detach().cpu().numpy() > 0)[0]
                
                if len(pred_inliers) > 3:
                    corr_idx = np.vstack((np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1]))).T
                    corr_idx = corr_idx[pred_inliers]
                    corr_vec = o3d.utility.Vector2iVector(corr_idx)
                    
                    reg_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                        src_pcd, tgt_pcd, corr_vec,
                        max_correspondence_distance=config.inlier_threshold,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        ransac_n=3,
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=50000)
                    )
                    pred_trans = torch.eye(4)[None].to(src_keypts.device)
                    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)

            if args.use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, src_normal, tgt_normal, pred_trans, config.inlier_threshold)

            # Metrics
            # Pass logits to ClassificationLoss (which uses BCEWithLogitsLoss)
            class_stats = class_loss(logits, gt_labels) 
            recall_val, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, confidence)

            # Stats recording
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

    return final_poses, stats, seed_num, seed_precision, correct_num, correct_ratio


def eval_3DMatch(model, config, args):
    """
    Collect the evaluation results on each scene of 3DMatch testset, write the result to a .log file.
    """
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    all_stats = {}
    all_poses = None
    avg_seed_num = 0
    avg_seed_precision = 0
    avg_correct_num = 0
    avg_correct_ratio = 0
    
    os.makedirs('logs/3dmatch', exist_ok=True)
    
    for scene_ind, scene in enumerate(scene_list):
        dset = ThreeDMatchTest(root=config.root,
                               descriptor=args.descriptor,
                               inlier_threshold=config.inlier_threshold,
                               num_node=args.num_points,
                               augment_axis=0,
                               augment_rotation=0.00,
                               augment_translation=0.0,
                               select_scene=scene,
                               )
        dloader = get_dataloader(dset, batch_size=1, num_workers=8, shuffle=False)
        scene_poses, scene_stats, seed_num, seed_precision, correct_num, correct_ratio = eval_3DMatch_scene(model, scene, scene_ind, dloader, config, args)
        
        avg_seed_num += seed_num
        avg_seed_precision += seed_precision
        avg_correct_num += correct_num
        avg_correct_ratio += correct_ratio
        
        if scene_ind == 0:
            all_poses = scene_poses
        else:
            all_poses = np.concatenate([all_poses, scene_poses], axis=0)
        all_stats[scene] = scene_stats

    # Result logging
    scene_vals = np.zeros([len(scene_list), 13])
    scene_ind = 0
    for scene, stats in all_stats.items():
        correct_pair = np.where(stats[:, 0] == 1)
        scene_vals[scene_ind] = stats.mean(0)
        if len(correct_pair[0]) > 0:
            scene_vals[scene_ind, 1] = stats[correct_pair].mean(0)[1]
            scene_vals[scene_ind, 2] = stats[correct_pair].mean(0)[2]
        logging.info(f"Scene {scene_ind}th: Reg Recall={scene_vals[scene_ind, 0] * 100:.2f}%")
        scene_ind += 1

    average = scene_vals.mean(0)
    logging.info(f"All {len(scene_list)} scenes, Mean Reg Recall={average[0] * 100:.2f}%")

    all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)
    return all_stats_npy



if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='TransformerPruner_3DMatch', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--descriptor', default='fcgf', type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--num_points', default='all', type=str)
    parser.add_argument('--use_icp', default=False, type=str2bool)
    args = parser.parse_args()
    
    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    config.inlier_threshold = 0.1
    config.re_thre = 15
    config.te_thre = 30

    if args.use_icp:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}-ICP.log'
    else:
        log_filename = f'logs/{args.chosen_snapshot}-{args.solver}-{config.descriptor}.log'
        
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    config.mode = "test"
    
    from models.outlier_pruner import MethodName
    model = MethodName(config)
    
    # Load Weights
    model_path = f'snapshot/{args.chosen_snapshot}/models/model_best.pkl'
    print(f"Loading model from {model_path}")
    miss = model.load_state_dict(torch.load(model_path), strict=True)
    print(miss)

    model.eval()
    
    # Calculate Parameters
    params = list(model.parameters())
    k = sum([p.numel() for p in params])
    print(f"Total Params: {k}")

    # Evaluate
    stats = eval_3DMatch(model.cuda(), config, args)

            

