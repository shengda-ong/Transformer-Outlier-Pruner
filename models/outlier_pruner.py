import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import rigid_transform_3d, knn

# vanila transformer pruner
class TransformerPruner(nn.Module):
    def __init__(self, in_dim=6, d_model=128, nhead=4, num_layers=6, dropout=0.1):
        super(TransformerPruner, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        encoded_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=d_model * 2,
                                                   dropout=dropout,
                                                   batch_first=True,
                                                   norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoded_layer,
                                                         num_layers=num_layers)
        # Head A: Classification (Inlier/Outlier)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        # Head B: Feature Projection for Subset Selection
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, corr_pos):
        '''
        [input]: corr_pos: (B, N, 6) tensor
        [output]: (B,N)
        '''
        B, N, _ = corr_pos.shape
        x = self.embedding(corr_pos.view(-1,6)) #(B*N,6)
        x = x.view(B,N,-1) # (B,N,128)
        x= self.transformer_encoder(x) #(B,N,128)

        # Confidence (Logits) for Classification
        logits = self.classifier(x).squeeze(-1) #(B,N)

        # Features for Subset Selection (k-NN)
        features = self.feature_proj(x)
        features = F.normalize(features, p=2, dim=-1) # L2 Normalize

        return logits, features

class MethodName(nn.Module):
    def __init__(self, config):
        super(MethodName, self).__init__()
        self.config = config
        self.pruner = TransformerPruner(
            in_dim=config.in_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        self.inlier_threshold = config.inlier_threshold
    
    def forward(self, input_data):
        '''
        Input Data Keys:
            - 'corr_pos': [B, N, 6] (Initial correspondences)
            - 'src_keypts': [B, N, 3]
            - 'tgt_keypts': [B, N, 3]
        '''
        corr = input_data['corr_pos']
        src_pts = input_data['src_keypts']
        tgt_pts = input_data['tgt_keypts']

        logits, features = self.pruner(corr)
        confidence = torch.sigmoid(logits)

        pred_trans = rigid_transform_3d(src_pts, tgt_pts, weights=confidence)
        res = {
            'final_trans': pred_trans,
            'logits': logits,
            'confidence': confidence,
            'features': features
        }

        return res

    def predict_hypotheses(self, input_data, num_seeds=100, k=20):
        '''
        [Inference Only]
        Implements the Seed -> Subset -> SVD pipeline using Transformer features.
        Replaces the Hypergraph lookup with k-NN
        '''

        res = self.forward(input_data)
        features = res['features'] # (B, N, D)
        confidence = res['confidence'] # (B, N)
        src_pts = input_data['src_keypts']
        tgt_pts = input_data['tgt_keypts']

        B, N, _ = src_pts.shape

        neighbor_indices = knn(features, k=k, ignore_self=True, normalized=True) # (B, N, k)

        actual_num_seeds = min(num_seeds, N)
        _, seed_indices = torch.topk(confidence, k=actual_num_seeds, dim=1) # (B, num_seeds)

        batch_hypotheses = []

        for b in range(B):
            #  Extract data for this batch
            b_seed_indices = seed_indices[b]
            b_neighbor_indices = neighbor_indices[b]
            b_src_pts = src_pts[b]
            b_tgt_pts = tgt_pts[b]

            # For each seed, look up its k neighbors
            subset_indices = b_neighbor_indices[b_seed_indices] # (num_seeds, k)

            # Gather the 3D coordinates for these subsets
            src_subsets = b_src_pts[subset_indices]
            tgt_subsets = b_tgt_pts[subset_indices]

            # SVD (Hypothesis Generation)
            hypotheses = rigid_transform_3d(src_subsets, tgt_subsets, weights=None)
            batch_hypotheses.append(hypotheses)
        
        if B == 1:
            return batch_hypotheses[0].unsqueeze(0)
        else:
            return torch.stack(batch_hypotheses)
        












