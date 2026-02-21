import torch
import torch.distributed as dist
import time, os
import numpy as np
from tensorboardX import SummaryWriter
from utils.timer import Timer, AverageMeter
from tqdm import tqdm

class Trainer(object):
    def __init__(self, args):
        # parameters
        self.t = None
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.batch_size = args.batch_size
        self.snapshot_dir = args.snapshot_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.evaluate_metric = args.evaluate_metric
        self.metric_weight = args.metric_weight
        self.transformation_loss_start_epoch = args.transformation_loss_start_epoch
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        # Curriculum state
        self.consensus_phase = False
        self.best_val_f1 = 0.0
        self.plateau_count = 0
        self.min_warmup = 10
        self.max_warmup = 20

        if self.gpu_mode:
            self.model = self.model.cuda()

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

    def train(self, resume, start_epoch, best_reg_recall, best_F1, consensus_phase=False):
        # resume to train from given epoch
        if resume:
            print('Resuming from epoch {}'.format(start_epoch))
            model_path = str(self.save_dir + '/model_{}.pkl'.format(start_epoch))
            print('Loading model parameters from {}'.format(model_path))
            self.model.load_state_dict(torch.load(model_path))
            
            # --- PHASE RESTORATION ---
            self.consensus_phase = consensus_phase
            # Auto-inference if epoch is large
            if start_epoch >= self.max_warmup:
                self.consensus_phase = True
            
            print(f">>> Resuming in {'CONSENSUS' if self.consensus_phase else 'WARMUP'} PHASE")
        else:
            start_epoch = 0
            best_reg_recall = 0
            best_F1 = 0
            self.consensus_phase = False
            print('Warning: Retrain the model may not produce the same results!')

        self.model.train()
        
        # Initial evaluation
        res = self.evaluate(start_epoch)
        print(f'Evaluation: Epoch {start_epoch}: SM Loss {res["sm_loss"]:.2f} '
              f'Class Loss {res["class_loss"]:.2f} F1 {res["f1"]:.2f} '
              f'Recall {res["reg_recall"]:.2f}')
              
        print('training start!!')
        self.t = tqdm(range(start_epoch, self.max_epoch), desc="Total Progress", ncols=100)
        
        for epoch_idx in self.t:
            epoch = epoch_idx + 1
            self.train_epoch(epoch)  # start from epoch 1
            
            # Validation
            if epoch % self.evaluate_interval == 0 or epoch == 1:
                res = self.evaluate(epoch)
                self.t.write(f'Evaluation: Epoch {epoch}: SM Loss {res["sm_loss"]:.2f} '
                             f'Class Loss {res["class_loss"]:.2f} F1 {res["f1"]:.2f} '
                             f'Recall {res["reg_recall"]:.2f} (Phase: {"Consensus" if self.consensus_phase else "Warmup"})')
                
                # --- Plateau Check for Curriculum Switch ---
                if not self.consensus_phase and epoch >= self.min_warmup:
                    f1_delta = res['f1'] - self.best_val_f1
                    if f1_delta < 0.01:
                        self.plateau_count += 1
                    else:
                        self.best_val_f1 = res['f1']
                        self.plateau_count = 0
                    
                    if self.plateau_count >= 3 or epoch >= self.max_warmup:
                        self.consensus_phase = True
                        self.t.write(f">>> [Epoch {epoch}] Plateau detected. Switching to CONSENSUS PHASE.")
                
                # --- Save Best Model ---
                if round(res['reg_recall'], 2) > best_reg_recall:
                    if epoch < 10:
                        self.t.write('best model in 10 epoch will not be saved!')
                    else:
                        best_reg_recall = round(res['reg_recall'], 2)
                        best_F1 = res['f1']
                        self._snapshot('best')
                elif round(res['reg_recall'], 2) == best_reg_recall and res['f1'] > best_F1:
                    self.t.write(f'previous best: RR {best_reg_recall:.2f} F1 {best_F1:.2f}, '
                                 f'current: RR {res["reg_recall"]:.2f} F1 {res["f1"]:.2f}')
                    if epoch < 10:
                        self.t.write('best model in 10 epoch will not be saved!')
                    else:
                        best_F1 = res['f1']
                        self._snapshot('best')

            if epoch % self.scheduler_interval == 0:
                self.scheduler.step()

            if epoch % self.snapshot_interval == 0:
                self._snapshot(epoch)

        self.t.write("Training finish!... save training results")
    
    def train_epoch(self, epoch):
        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        if self.consensus_phase:
            meter_list += ['edge_loss', 'topk_loss']
            
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.train_loader.dataset) / self.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        
        trainer_loader_iter = iter(self.train_loader)
        
        for iter_idx in range(num_iter):
            data_timer.tic()
            try:
                batch = next(trainer_loader_iter)
            except StopIteration:
                trainer_loader_iter = iter(self.train_loader)
                batch = next(trainer_loader_iter)
                
            (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) = batch
            
            if self.gpu_mode:
                corr_pos = corr_pos.cuda()
                src_keypts = src_keypts.cuda()
                tgt_keypts = tgt_keypts.cuda()
                src_normal = src_normal.cuda()
                tgt_normal = tgt_normal.cuda()
                gt_trans = gt_trans.cuda()
                gt_labels = gt_labels.cuda()

            # Data filtering strategy
            if epoch <= 5:
                mask = gt_labels.mean(-1) > 0.2
                if mask.sum() > 0:
                    corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels = \
                        corr_pos[mask], src_keypts[mask], tgt_keypts[mask], src_normal[mask], tgt_normal[mask], gt_trans[mask], gt_labels[mask]
            elif epoch <= 10:
                mask = gt_labels.mean(-1) > 0.1
                if mask.sum() > 0:
                    corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels = \
                        corr_pos[mask], src_keypts[mask], tgt_keypts[mask], src_normal[mask], tgt_normal[mask], gt_trans[mask], gt_labels[mask]

            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'src_normal': src_normal,
                'tgt_normal': tgt_normal,
            }
            data_timer.toc()

            model_timer.tic()
            self.optimizer.zero_grad()
            
            res = self.model(data)
            
            # --- BASE LOSSES ---
            class_stats = self.evaluate_metric['ClassificationLoss'](res['logits'], gt_labels)
            class_loss = class_stats['loss']
            
            M = torch.matmul(res['features'], res['features'].transpose(1, 2))
            M = torch.clamp(M, min=0, max=1)
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](M, gt_labels)

            total_loss = (self.metric_weight['ClassificationLoss'] * class_loss + 
                          self.metric_weight['SpectralMatchingLoss'] * sm_loss)
            
            # --- CONSENSUS PHASE LOSSES ---
            edge_loss_val, topk_loss_val = 0.0, 0.0
            if self.consensus_phase:
                edge_loss_val = self.evaluate_metric['EdgeFeatureLoss'](res['attn_weights'], gt_labels)
                topk_loss_val = self.evaluate_metric['TopKClassificationLoss'](res['logits'], gt_labels)
                total_loss += (self.metric_weight['EdgeFeatureLoss'] * edge_loss_val + 
                               self.metric_weight['TopKClassificationLoss'] * topk_loss_val)

            # Global RR for logging
            reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](
                res['final_trans'], gt_trans, src_keypts, tgt_keypts, res['confidence']
            )

            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(te),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }
            if self.consensus_phase:
                stats['edge_loss'] = float(edge_loss_val)
                stats['topk_loss'] = float(topk_loss_val)

            total_loss.backward()
            
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False; break
            if do_step: self.optimizer.step()
            model_timer.toc()

            if not np.isnan(float(total_loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]): meter_dict[key].update(stats[key])
            
            if (iter_idx + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter_idx
                for key in meter_list: self.writer.add_scalar(f"Train/{key}", meter_dict[key].avg, curr_iter)
                
                log_str = f"Epoch: {epoch} [{iter_idx + 1:4d}/{num_iter}] sm: {meter_dict['sm_loss'].avg:.2f} cl: {meter_dict['class_loss'].avg:.2f} RR: {meter_dict['reg_recall'].avg:.2f}% "
                if self.consensus_phase:
                    log_str += f"edge: {meter_dict['edge_loss'].avg:.4f} topk: {meter_dict['topk_loss'].avg:.2f} "
                self.t.write(log_str)
                
    def evaluate(self, epoch):
        self.model.eval()
        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 'f1']
        meter_dict = {key: AverageMeter() for key in meter_list}

        num_iter = int(len(self.val_loader.dataset) / self.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = iter(self.val_loader)
        
        with torch.no_grad():
            for iter_idx in range(num_iter):
                try: batch = next(val_loader_iter)
                except StopIteration: val_loader_iter = iter(self.val_loader); batch = next(val_loader_iter)
                    
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) = batch
                if self.gpu_mode:
                    corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                        corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
                    
                data = {'corr_pos': corr_pos, 'src_keypts': src_keypts, 'tgt_keypts': tgt_keypts}
                res = self.model(data)
                
                # Metrics
                class_stats = self.evaluate_metric['ClassificationLoss'](res['logits'], gt_labels)
                
                # --- RR Metric Selection ---
                if self.consensus_phase:
                    # During Consensus Phase, use actual SAMPLING RECALL for saving best model
                    sampled_trans = self.model.predict_hypotheses(data, num_seeds=100, k=20) # [B, num_hyp, 4, 4]
                    
                    # Pick best hypothesis for recall calculation (Batch size is usually 1 for val)
                    b_idx = 0
                    b_sampled = sampled_trans[b_idx] # [num_hyp, 4, 4]
                    num_hyp = b_sampled.shape[0]
                    
                    b_src = src_keypts[b_idx]
                    b_tgt = tgt_keypts[b_idx]
                    
                    src_expanded = b_src.unsqueeze(0).repeat(num_hyp, 1, 1) # [num_hyp, N, 3]
                    R_hyp = b_sampled[:, :3, :3]
                    t_hyp = b_sampled[:, :3, 3].unsqueeze(1)
                    
                    src_warped = torch.matmul(src_expanded, R_hyp.transpose(1, 2)) + t_hyp
                    dists = torch.norm(src_warped - b_tgt.unsqueeze(0), dim=-1)
                    inliers = (dists < 0.1).sum(dim=1)
                    best_hyp_idx = torch.argmax(inliers)
                    final_trans = b_sampled[best_hyp_idx].unsqueeze(0) # [1, 4, 4]
                    
                    reg_recall, _, _, _ = self.evaluate_metric['TransformationLoss'](final_trans, gt_trans[b_idx:b_idx+1], src_keypts[b_idx:b_idx+1], tgt_keypts[b_idx:b_idx+1])
                else:
                    # During Warmup, use fast Global SVD Recall
                    reg_recall, _, _, _ = self.evaluate_metric['TransformationLoss'](res['final_trans'], gt_trans, src_keypts, tgt_keypts)

                # --- Accurate SM Loss for Validation Logs ---
                M_val = torch.matmul(res['features'], res['features'].transpose(1, 2))
                M_val = torch.clamp(M_val, min=0, max=1)
                sm_loss_val = self.evaluate_metric['SpectralMatchingLoss'](M_val, gt_labels)

                meter_dict['class_loss'].update(float(class_stats['loss']))
                meter_dict['sm_loss'].update(float(sm_loss_val))
                meter_dict['reg_recall'].update(float(reg_recall))
                meter_dict['f1'].update(class_stats['f1'])
                
        res = {k: v.avg for k, v in meter_dict.items()}
        for key in meter_list: self.writer.add_scalar(f"Val/{key}", res[key], epoch)
        self.model.train()
        return res

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))
        msg = f"Save model to {self.save_dir}/model_{epoch}.pkl"
        if self.t is not None:
            self.t.write(msg)
        else:
            print(msg)
    
    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        msg = f"Load model from {pretrain}.pkl"
        if self.t is not None:
            self.t.write(msg)
        else:
            print(msg)