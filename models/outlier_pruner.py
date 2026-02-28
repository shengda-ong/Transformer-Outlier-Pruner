import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import rigid_transform_3d, knn
from einops import rearrange

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
        x = src
        if self.norm_first:
            x_norm = self.norm1(x)
            src_mixed = torch.bmm(sog_matrix, x_norm)
            attn_out, attn_weights = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
        else:
            src_mixed = torch.bmm(sog_matrix, x)
            attn_out, attn_weights = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._ff_block(x))
        
        return x, attn_weights

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x, weights = self.self_attn(x, x, x,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=True,
                                   is_causal=is_causal,
                                   average_attn_weights=False)
        return x, weights
    
 
class CAAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super(CAAttention, self).__init__()
        self.heads = heads
        # Learnable temperature parameter for channel attention
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        # 1x1 Convolutions act as point-wise linear layers across channels
        self.query_filter = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.key_filter = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.value_filter = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        # Input x: (B, N, C) -> Reshape to (B, C, N, 1) for Conv2d
        x1 = x.transpose(1, 2).unsqueeze(-1)
        B, C, N, _ = x1.shape
        q = self.query_filter(x1)
        k = self.key_filter(x1)
        v = self.value_filter(x1)

        # Multi-head attention across channels
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=N, w=1)
        out = self.project_out(out)
        
        # Back to (B, N, C)
        out = out.squeeze(-1).transpose(1, 2)
        return out + x # Residual connection inside CA


class SACAMixingLayer(SOGMixingLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first, norm_first):
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                         batch_first=batch_first, norm_first=norm_first)
        # Initialize the Channel Attention block
        self.channel_attn = CAAttention(channels=d_model, heads=nhead)

    def forward(self, src, sog_matrix, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
             # Part 1: Spatial Attention (with SOG Mixing)
            x_norm = self.norm1(x)
            src_mixed = torch.bmm(sog_matrix, x_norm)
            attn_out, attn_weights = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = x + attn_out

            # Part 2: Channel Attention and Feed-Forward
            x_norm2 = self.norm2(x)
            x_ca = self.channel_attn(x_norm2) # Enhance features with channel context
            x = x + self._ff_block(x_ca)
        else:
            # Standard implementation for post-norm
            src_mixed = torch.bmm(sog_matrix, x)
            attn_out, attn_weights = self._sa_block(src_mixed, src_mask, src_key_padding_mask, is_causal)
            x = self.norm1(x + attn_out)

            x_ca = self.channel_attn(x)
            x = self.norm2(x + self._ff_block(x_ca))
        
        return x, attn_weights
            



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

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, sog_matrix) #(B,N,128)
            all_attn_weights.append(attn_weights)
        
        logits = self.classifier(x).squeeze(-1)
        features = self.feature_proj(x)
        features = F.normalize(features, p=2, dim=-1)

        return logits, features, all_attn_weights


class TransformerGeoPrunerCA(TransformerGeoPruner):
    def __init__(self, in_dim=6, d_model=128, nhead=4, num_layers=10, dropout=0.1, sigma_d=0.1):
        super().__init__(in_dim, d_model, nhead, num_layers, dropout, sigma_d)

        # Replace baseline layers with CA-enabled layers
        # All parameters (d_model, nhead, dim_feedforward=d_model*2) remain identical

        self.layers = nn.ModuleList([
            SACAMixingLayer(d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=d_model * 2,
                            dropout=dropout,
                            batch_first=True,
                            norm_first=True)
            for _ in range(num_layers)
        ])

class MethodName(nn.Module):
    def __init__(self, config):
        super(MethodName, self).__init__()
        self.config = config
        self.pruner = TransformerGeoPrunerCA(
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

        logits, features, attn_weights = self.pruner(corr)
        confidence = torch.sigmoid(logits)

        pred_trans = rigid_transform_3d(src_pts, tgt_pts, weights=confidence)
        
        # Select top-k seeds based on confidence for logging/testing
        _, topk_seeds = torch.topk(confidence, k=min(100, confidence.shape[1]), dim=1)

        res = {
            'final_trans': pred_trans,
            'logits': logits,
            'confidence': confidence,
            'features': features,
            'attn_weights': attn_weights,
            'seeds': topk_seeds
        }

        return res

    def cal_leading_eigenvector(self, M, num_iterations=10):
        """
        Power iteration to find the leading eigenvector (weights) of compatibility matrix M.
        M: [num_seeds, k, k]
        """
        leading_eig = torch.ones_like(M[:, :, 0:1])
        for _ in range(num_iterations):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        return leading_eig.squeeze(-1)

    def spatial_nms(self, src_pts, confidence, radius):
        """
        Perform Spatial NMS on source keypoints to diversify seeds.
        src_pts: [N, 3], confidence: [N]
        """
        indices = torch.argsort(confidence, descending=True)
        keep = []
        while len(indices) > 0:
            idx = indices[0]
            keep.append(idx.item())
            if len(indices) == 1: break
            
            dist = torch.norm(src_pts[indices[1:]] - src_pts[idx], dim=-1)
            mask = dist > radius
            indices = indices[1:][mask]
        return torch.tensor(keep, device=src_pts.device)

    def predict_hypotheses(self, input_data, num_seeds=100, k=20):
        '''
        [Inference Only] Robust Hypothesis Generation
        Mimics HyperGCT: Spatial NMS -> Local Compatibility -> Power Iteration -> SVD
        '''
        res = self.forward(input_data)
        features = res['features'] # (B, N, D)
        confidence = res['confidence'] # (B, N)
        src_pts = input_data['src_keypts']
        tgt_pts = input_data['tgt_keypts']
        B, N, _ = src_pts.shape

        batch_hypotheses = []
        for b in range(B):
            # 1. Spatial NMS to get diverse seeds
            b_src = src_pts[b]
            b_tgt = tgt_pts[b]
            b_conf = confidence[b]
            b_feat = features[b]
            
            # Use inlier_threshold as NMS radius (diversify seeds)
            seed_indices = self.spatial_nms(b_src, b_conf, self.inlier_threshold)
            seed_indices = seed_indices[:num_seeds]
            
            # 2. Find k-NN in feature space for each seed
            # neighbor_indices: (num_seeds, k)
            dist_feat = torch.cdist(b_feat[seed_indices], b_feat)
            neighbor_indices = torch.topk(dist_feat, k=k, largest=False)[1]

            # 3. Build local compatibility matrices for subsets
            # Gather subsets: (num_seeds, k, 3)
            src_sub = b_src[neighbor_indices]
            tgt_sub = b_tgt[neighbor_indices]
            feat_sub = b_feat[neighbor_indices]

            # Spatial consistency (length preservation): (num_seeds, k, k)
            dist_s = torch.cdist(src_sub, src_sub)
            dist_t = torch.cdist(tgt_sub, tgt_sub)
            M_spatial = torch.clamp(1 - (dist_s - dist_t)**2 / self.inlier_threshold**2, min=0)

            # Feature consistency: (num_seeds, k, k)
            M_feat = torch.bmm(feat_sub, feat_sub.transpose(1, 2))
            M_feat = torch.clamp(M_feat, min=0)

            # Total compatibility: (num_seeds, k, k)
            M = M_spatial * M_feat
            M[:, torch.arange(k), torch.arange(k)] = 0 # zero diagonal

            # 4. Power Iteration to get weights
            weights = self.cal_leading_eigenvector(M) # (num_seeds, k)

            # 5. Weighted SVD for each seed's subset
            hypotheses = rigid_transform_3d(src_sub, tgt_sub, weights=weights)
            batch_hypotheses.append(hypotheses)
        
        return torch.stack(batch_hypotheses)
        












