import os
import sys
import time
import shutil
import json

# Ensure the current directory is in the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from easydict import EasyDict as edict
from libs.loss import *
from datasets.ThreeDMatch import ThreeDMatchTrainVal
from datasets.dataloader import get_dataloader_train
from libs.trainer import Trainer
from torch import optim
import torch

if __name__ == '__main__':
    config = get_config()
    dconfig = vars(config)
    for k in dconfig:
        print(f"    {k}: {dconfig[k]}")
    config = edict(dconfig)

    resume = False
    if resume:
        start_epoch = 0 # Update this if resuming
        best_reg_recall = 0
        best_F1 = 0
        consensus_phase = False # Update this if resuming in phase 2
        experiment_id = config.exp_id # Use ID from config
        config.snapshot_dir = f'snapshot/{experiment_id}'
        config.tboard_dir = f'tensorboard/{experiment_id}'
        config.save_dir = os.path.join(f'snapshot/{experiment_id}', 'models/')
    else:
        start_epoch = 0
        best_reg_recall = 0
        best_F1 = 0
        consensus_phase = False

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    shutil.copy2(os.path.join('.', 'train_3DMatch.py'), os.path.join(config.snapshot_dir, 'train.py'))
    shutil.copy2(os.path.join('.', 'libs/trainer.py'), os.path.join(config.snapshot_dir, 'trainer.py'))
    # Copy your new model file
    shutil.copy2(os.path.join('.', 'models/outlier_pruner.py'), os.path.join(config.snapshot_dir, 'outlier_pruner.py'))  
    shutil.copy2(os.path.join('.', 'libs/loss.py'), os.path.join(config.snapshot_dir, 'loss.py'))
    shutil.copy2(os.path.join('.', 'datasets/ThreeDMatch.py'), os.path.join(config.snapshot_dir, 'dataset.py'))
    shutil.copy2(os.path.join('.', 'datasets/dataloader.py'), os.path.join(config.snapshot_dir, 'loader.py'))

    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )

    from models.outlier_pruner import MethodName

    config.mode = "train"
    config.model = MethodName(config)

    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            # momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    train_set = ThreeDMatchTrainVal(root=config.root, 
                        descriptor=config.descriptor,
                        split='train', 
                        inlier_threshold=config.inlier_threshold,
                        num_node=config.num_node, 
                        downsample=config.downsample,
                        augment_axis=config.augment_axis,
                        augment_rotation=config.augment_rotation,
                        augment_translation=config.augment_translation,
                        )
    val_set = ThreeDMatchTrainVal(root=config.root,
                            split='val',
                            descriptor=config.descriptor,
                            inlier_threshold=config.inlier_threshold,
                            num_node=config.num_node,
                            downsample=config.downsample,
                            augment_axis=config.augment_axis,
                            augment_rotation=config.augment_rotation,
                            augment_translation=config.augment_translation,
                            )
    config.train_loader = get_dataloader_train(dataset=train_set, 
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader = get_dataloader_train(dataset=val_set,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        )
    
    config.evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=config.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=config.balanced),
        "TransformationLoss": TransformationLoss(re_thresh=config.re_thre, te_thresh=config.te_thre),
        "EdgeFeatureLoss": EdgeFeatureLoss(),
        "TopKClassificationLoss": TopKClassificationLoss(top_k=100),
    }

    config.metric_weight = {
        "ClassificationLoss": config.weight_classification,
        "SpectralMatchingLoss": config.weight_spectralmatching,
        "TransformationLoss": config.weight_transformation,
        "EdgeFeatureLoss": config.weight_edgefeature, # Target weight for Phase 2
        "TopKClassificationLoss": config.weight_topk, # Target weight for Phase 2
    }

    params = list(config.model.parameters())
    k = 0
    for param in params:
        l = 1
        for j in param.size():
            l *= j
        k = k + l
    print(f"Total Parameters: {k}")

    trainer = Trainer(config)
    trainer.train(resume, start_epoch, best_reg_recall, best_F1, consensus_phase)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Release trainer
    del trainer