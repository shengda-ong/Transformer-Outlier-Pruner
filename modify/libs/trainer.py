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

        if self.gpu_mode:
            self.model = self.model.cuda()

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

    def train(self, resume, start_epoch, best_reg_recall, best_F1):
        # resume to train from given epoch
        if resume:
            print('Resuming from epoch {}'.format(start_epoch))
            model_path = str(self.save_dir + '/model_{}.pkl'.format(start_epoch))
            print('Loading model parameters from {}'.format(model_path))
            self.model.load_state_dict(torch.load(model_path))
        else:
            start_epoch = 0
            best_reg_recall = 0
            best_F1 = 0
            print('Warning: Retrain the model may not produce the same results!')

        self.model.train()
        
        # Initial evaluation
        res = self.evaluate(start_epoch)
        print(f'Evaluation: Epoch {start_epoch}: SM Loss {res["sm_loss"]:.2f} '
              f'Class Loss {res["class_loss"]:.2f} F1 {res["f1"]:.2f} '
              f'Recall {res["reg_recall"]:.2f}')
              
        print('training start!!')
        self.t = tqdm(range(start_epoch, self.max_epoch), desc="Total Progress", ncols=100)
        
        for epoch in self.t:
            self.train_epoch(epoch + 1)  # start from epoch 1
            
            # Validation
            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                self.t.write(f'Evaluation: Epoch {epoch + 1}: SM Loss {res["sm_loss"]:.2f} '
                             f'Class Loss {res["class_loss"]:.2f} F1 {res["f1"]:.2f} '
                             f'Recall {res["reg_recall"]:.2f}')
                             
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

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        self.t.write("Training finish!... save training results")
    
    def train_epoch(self, epoch):
        # Removed graph_loss from meters
        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.train_loader.dataset) / self.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        
        # Use simple iteration for robustness on 1 GPU
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

            # Data filtering strategy (Curriculum learning)
            if epoch <= 5:
                mask = gt_labels.mean(-1) > 0.2
                if mask.sum() > 0:
                    corr_pos = corr_pos[mask]
                    src_keypts = src_keypts[mask]
                    tgt_keypts = tgt_keypts[mask]
                    src_normal = src_normal[mask]
                    tgt_normal = tgt_normal[mask]
                    gt_trans = gt_trans[mask]
                    gt_labels = gt_labels[mask]

            elif epoch <= 10:
                mask = gt_labels.mean(-1) > 0.1
                if mask.sum() > 0:
                    corr_pos = corr_pos[mask]
                    src_keypts = src_keypts[mask]
                    tgt_keypts = tgt_keypts[mask]
                    src_normal = src_normal[mask]
                    tgt_normal = tgt_normal[mask]
                    gt_trans = gt_trans[mask]
                    gt_labels = gt_labels[mask]

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
            
            # Forward Pass
            res = self.model(data)
            
            # Extract outputs
            pred_trans = res['final_trans']
            logits = res['logits']          # (B, N) Logits for BCE
            features = res['features']      # (B, N, D) Normalized Features
            confidence = res['confidence']  # (B, N) Probs [0,1]
            
            # --- LOSS CALCULATION ---
            
            # 1. Classification Loss (BCEWithLogits)
            class_stats = self.evaluate_metric['ClassificationLoss'](logits, gt_labels)
            class_loss = class_stats['loss']
            
            # 2. Spectral Matching Loss
            # Compute pairwise compatibility matrix M from features
            # M = features @ features.T. Since features are normalized, this is cosine similarity.
            # We clamp to [0,1] because SpectralMatchingLoss expects positive compatibility.
            M = torch.matmul(features, features.transpose(1, 2))
            M = torch.clamp(M, min=0, max=1)
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](M, gt_labels)

            # 3. Transformation Loss
            reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](
                pred_trans, gt_trans, src_keypts, tgt_keypts, confidence
            )

            # Total Loss (Weighted Sum)
            loss = (self.metric_weight['ClassificationLoss'] * class_loss + 
                    self.metric_weight['SpectralMatchingLoss'] * sm_loss)
            

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

            # Backward
            loss.backward()
            
            # Gradient Clipping / Check
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step:
                self.optimizer.step()
            model_timer.toc()

            # Logging
            if not np.isnan(float(loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])
            else:
                import pdb; pdb.set_trace()

            if (iter_idx + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter_idx
                for key in meter_list:
                    self.writer.add_scalar(f"Train/{key}", meter_dict[key].avg, curr_iter)

                self.t.write(f"Epoch: {epoch} [{iter_idx + 1:4d}/{num_iter}] "
                             f"sm_loss: {meter_dict['sm_loss'].avg:.2f} "
                             f"class_loss: {meter_dict['class_loss'].avg:.2f} "
                             f"reg_recall: {meter_dict['reg_recall'].avg:.2f}% "
                             f"re: {meter_dict['re'].avg:.2f} "
                             f"te: {meter_dict['te'].avg:.2f} "
                             f"data_time: {data_timer.avg:.2f}s "
                             f"model_time: {model_timer.avg:.2f}s "
                             )
                
    def evaluate(self, epoch):
        self.model.eval()

        meter_list = ['class_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.val_loader.dataset) / self.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = iter(self.val_loader)
        
        with torch.no_grad():
            for iter_idx in range(num_iter):
                data_timer.tic()
                try:
                    batch = next(val_loader_iter)
                except StopIteration:
                    val_loader_iter = iter(self.val_loader)
                    batch = next(val_loader_iter)
                    
                (corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal, gt_trans, gt_labels) = batch
                
                if self.gpu_mode:
                    corr_pos = corr_pos.cuda()
                    src_keypts = src_keypts.cuda()
                    tgt_keypts = tgt_keypts.cuda()
                    src_normal = src_normal.cuda()
                    tgt_normal = tgt_normal.cuda()
                    gt_trans = gt_trans.cuda()
                    gt_labels = gt_labels.cuda()
                    
                data = {
                    'corr_pos': corr_pos,
                    'src_keypts': src_keypts,
                    'tgt_keypts': tgt_keypts,
                    'src_normal': src_normal,
                    'tgt_normal': tgt_normal,
                }
                data_timer.toc()

                model_timer.tic()
                # Forward
                res = self.model(data)
                
                # Extract
                pred_trans = res['final_trans']
                logits = res['logits']
                features = res['features']
                confidence = res['confidence']

                # 1. Class Loss
                class_stats = self.evaluate_metric['ClassificationLoss'](logits, gt_labels)
                class_loss = class_stats['loss']
                
                # 2. SM Loss (Compute M)
                M = torch.matmul(features, features.transpose(1, 2))
                M = torch.clamp(M, min=0, max=1)
                sm_loss = self.evaluate_metric['SpectralMatchingLoss'](M, gt_labels)

                # 3. Transform Loss
                reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](
                    pred_trans, gt_trans, src_keypts, tgt_keypts, confidence
                )
                
                model_timer.toc()

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
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])
                torch.cuda.empty_cache()

        res = {
            'sm_loss': meter_dict['sm_loss'].avg,
            'class_loss': meter_dict['class_loss'].avg,
            'reg_recall': meter_dict['reg_recall'].avg,
            'f1': meter_dict['f1'].avg,
        }
        
        # Log validation metrics
        for key in meter_list:
            self.writer.add_scalar(f"Val/{key}", meter_dict[key].avg, epoch)

        return res

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))
        self.t.write(f"Save model to {self.save_dir}/model_{epoch}.pkl")
    
    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.t.write(f"Load model from {pretrain}.pkl")