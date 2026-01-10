import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from utils.SE3 import *
import warnings

warnings.filterwarnings('ignore')


def transformation_error(pred_trans, gt_trans):
    """
    Computes rotation and translation errors between predicted and ground truth transformations.
    """
    if len(pred_trans.shape) == 3:
        bs = pred_trans.shape[0]
        pred_Rs = pred_trans[:, :3, :3]
        gt_Rs = gt_trans[:, :3, :3]
        pred_ts = pred_trans[:, :3, 3:4]
        gt_ts = gt_trans[:, :3, 3:4]
        
        # Calculate RE (Rotation Error)
        mat = torch.matmul(pred_Rs.transpose(-1, -2), gt_Rs)
        tr = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
        RE = torch.acos(torch.clamp(0.5 * (tr - 1.0), min=-1, max=1)) * 180 / np.pi
        
        # Calculate TE (Translation Error)
        TE = torch.norm(pred_ts - gt_ts, dim=1) * 100
        
        RE = RE.reshape(bs)
        TE = TE.reshape(bs)

    else:
        pred_R = pred_trans[:3, :3]
        gt_R = gt_trans[:3, :3]
        pred_t = pred_trans[:3, 3:4]
        gt_t = gt_trans[:3, 3:4]
        tr = torch.trace(pred_R.T @ gt_R)
        RE = torch.acos(torch.clamp(0.5 * (tr - 1), min=-1, max=1)) * 180 / np.pi
        TE = torch.norm(pred_t - gt_t) * 100
    return RE, TE

class TransformationLoss(nn.Module):
    """
    Measures the success of the registration.
    Used for evaluation
    """
    def __init__(self, re_thresh, te_thresh):
        super(TransformationLoss, self).__init__()
        self.re_thresh = re_thresh
        self.te_thresh = te_thresh

    def forward(self, pred_trans, gt_trans, src_kpts, tgt_kpts, inlier_probs=None):
        batch_size = pred_trans.shape[0]
        
        # Compute Errors
        RE, TE = transformation_error(pred_trans, gt_trans)

        # Compute RMSE (Root Mean Square Error) of transformed points
        trans_src = transform(src_kpts, pred_trans)
        RMSE = torch.norm(trans_src - tgt_kpts, dim=-1).mean(axis=1).reshape(batch_size)
        
        # Calculate Recall (Success Rate)
        # Success if both RE and TE are below thresholds
        succ = (RE < self.re_thresh) & (TE < self.te_thresh)
        recall = succ.float().sum()

        return recall * 100.0 / batch_size, RE.mean(), TE.mean(), RMSE.mean()
    


class ClassificationLoss(nn.Module):
    """
    Standard Binary Cross Entropy for Inlier/Outlier classification.
    """
    def __init__(self, balanced=True):
        super(ClassificationLoss, self).__init__()
        self.balanced = balanced

    def forward(self, pred, gt, weight=None):
        """ 
        Inputs:
            - pred: [bs, num_corr] predicted logits
            - gt:   [bs, num_corr] ground truth labels (0 or 1)
        """
        bs, num_corr = pred.shape
        num_pos = torch.relu(torch.sum(gt) - 1) + 1
        num_neg = torch.relu(torch.sum(1 - gt) - 1) + 1
        
        if weight is not None:
            loss = nn.BCEWithLogitsLoss(reduction='none')(pred, gt.float()) 
            loss = torch.mean(loss * weight)
        elif self.balanced is False:
            loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt.float())
        else:
            # Handle class imbalance (usually far more outliers than inliers)
            pos_weight = num_neg * 1.0 / num_pos
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')(pred, gt.float())

        # Compute metrics for logging
        precision, recall, f1 = 0.0, 0.0, 0.0
        logit_true, logit_false = 0.0, 0.0
        pred_labels = pred > 0
        

        for i in range(bs):
            gt_np = gt[i].detach().cpu().numpy()
            pred_labels_np = pred_labels[i].detach().cpu().numpy()
            pred_np = pred[i].detach().cpu().numpy()
            
            precision += precision_score(gt_np, pred_labels_np, zero_division=0)
            recall += recall_score(gt_np, pred_labels_np, zero_division=0)
            f1 += f1_score(gt_np, pred_labels_np, zero_division=0)
            
            logit_true += np.sum(pred_np * gt_np) / max(1, np.sum(gt_np))
            logit_false += np.sum(pred_np * (1 - gt_np)) / max(1, np.sum(1 - gt_np))

        eval_stats = {
            "loss": loss,
            "precision": float(precision / bs),
            "recall": float(recall / bs),
            "f1": float(f1 / bs),
            "logit_true": float(logit_true / bs),
            "logit_false": float(logit_false / bs)
        }
        return eval_stats
    
class SpectralMatchingLoss(nn.Module):
    """
    Enforces geometric consistency on the learned features.
    If two correspondences are both inliers, their feature distance should correspond 
    to their spatial compatibility.
    """
    def __init__(self, balanced=True):
        super(SpectralMatchingLoss, self).__init__()
        self.balanced = balanced

    def forward(self, M, gt_labels):
        """ 
        Inputs:
            - M:    [bs, num_corr, num_corr] Pairwise Feature Compatibility Matrix.
                    (You must compute this from your Transformer features before passing here)
            - gt_labels:   [bs, num_corr] ground truth inlier/outlier labels
        """
        # Create Ground Truth Compatibility Matrix
        # Entry (i, j) is 1 iff both i and j are inliers
        gt_M = ((gt_labels[:, None, :] + gt_labels[:, :, None]) == 2)
        
        # Ignore self-loops (diagonal)
        for i in range(gt_M.shape[0]):
            gt_M[i].fill_diagonal_(0)
            
        if self.balanced:
            # Balanced MSE: Treats the (usually sparse) positive matches with higher weight
            sm_loss_p = ((M - 1) ** 2 * gt_M).sum(-1).sum(-1) / (torch.relu((gt_M).sum(-1).sum(-1) - 1.0) + 1.0)
            sm_loss_n = ((M - 0) ** 2 * (1 - gt_M)).sum(-1).sum(-1) / (torch.relu((1 - gt_M).sum(-1).sum(-1) - 1.0) + 1.0)
            loss = torch.mean(sm_loss_p * 0.5 + sm_loss_n * 0.5)
        else:
            loss = torch.nn.MSELoss(reduction='mean')(M, gt_M.float())
            
        return loss

