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


'''
geometry-aware transformer pruner
'''
# This layer accepts the SOG matrix and performs the mixing (SOG @ X)
class SOGMixingLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first, norm_first):
        super().__init__(d_model, nhead, dim_feedforward, dropout, 
                         batch_first=batch_first, norm_first=norm_first)
        
    def forward(self, src, sog_matrix, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src shape: (B, N, D)
        # sog_matrix shape: (B, N, N)
        # (B, N, N) @ (B, N, D) -> (B, N, D)

        x = src
        if self.norm_first:
            '''
            1. Normalize first
            2. Mix the normalized features (SOG @ X_norm)
            3. Attention
            4. Residual Connection
            5. FF & Norm
            '''          
            x_norm = self.norm1(x)
            src_mixed = torch.bmm(sog_matrix, x_norm)
            attn_out = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
        else:

            '''
            1. Mix raw features (SOG @ X)
            2. Attention
            3. Residual Connection & Norm
            4. FF & Norm
            '''
            src_mixed = torch.bmm(sog_matrix, x)
            attn_out = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._ff_block(x))
        
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        return self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False,
                              is_causal=is_causal)[0]


class TransformerGeoPruner(nn.Module):
    def __init__(self, in_dim=6, d_model=128, nhead=4, num_layers=10, dropout=0.1, sigma_d=0.1):
        super(TransformerGeoPruner, self).__init__()
        self.sigma_d = sigma_d

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.layers = nn.ModuleList([
            SOGMixingLayer(d_model=d_model, 
                           nhead=nhead,
                           dim_feedforward=d_model * 2,
                           dropout=dropout,
                           batch_first=True,
                           norm_first=True)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def compute_sog_matrix(self, corr_pos):
        B, N, _ = corr_pos.shape
        src_pts = corr_pos[:, :, :3]
        tgt_pts = corr_pos[:, :, 3:]

        '''
        1. Compute pairwise distance matrices for source and target points
        2. Compute Compatibiltiy scores
        3. W * (W X W)
        4. Row-normalize
        '''
        # distance matrices
        dist_s = torch.cdist(src_pts, src_pts)
        dist_t = torch.cdist(tgt_pts, tgt_pts)

        # compatibility scores
        d_ij = torch.abs(dist_s - dist_t)
        sog = torch.clamp(1 - (d_ij.pow(2) / (self.sigma_d ** 2)), min=0)

        # SOG matrix
        sog = sog * torch.bmm(sog, sog)

        # row-normalize
        row_sums = sog.sum(dim=2, keepdim=True) + 1e-6
        sog = sog / row_sums

        return sog
    
    def forward(self, corr_pos):
        B, N, _ = corr_pos.shape
        x = self.embedding(corr_pos.view(-1, 6)).view(B, N, -1) #(B,N,128)
        sog_matrix = self.compute_sog_matrix(corr_pos)

        for layer in self.layers:
            x = layer(x, sog_matrix) #(B,N,128)
        
        logits = self.classifier(x).squeeze(-1)
        features = self.feature_proj(x)
        features = F.normalize(features, p=2, dim=-1)

        return logits, features


class MethodName(nn.Module):
    def __init__(self, config):
        super(MethodName, self).__init__()
        self.config = config
        # self.pruner = TransformerPruner(
        #     in_dim=config.in_dim,
        #     d_model=config.d_model,
        #     nhead=config.nhead,
        #     num_layers=config.num_layers,
        #     dropout=config.dropout
        # )
        self.pruner = TransformerGeoPruner(
            in_dim=config.in_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            sigma_d=0.1 # according to paper
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
        












