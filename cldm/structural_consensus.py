import torch
from typing import List

class StructuralSemanticConsensus(torch.nn.Module):
    def __init__(self, clip_model, garment_segmentor,
                 lambda_cons: float = 0.5,
                 lambda_global: float = 0.5):
        super().__init__()
        self.clip_model = clip_model          # encode_text, encode_image
        self.garment_segmentor = garment_segmentor
        self.lambda_cons = lambda_cons
        self.lambda_global = lambda_global

    def forward(self, x_img: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        x_img: (B, 3, H, W) in [-1, 1]
        texts: list[str] original captions
        """
        if self.clip_model is None or self.garment_segmentor is None:
            return x_img.new_tensor(0.0)

        # 1) [-1,1] -> [0,1]
        x_for_seg = (x_img + 1.0) / 2.0

        # 2) Segment garment regions: (B, K, H, W)
        part_masks = self.garment_segmentor(x_for_seg)
        B, K, H, W = part_masks.shape

        # 3) CLIP image embedding for each part
        img_feats_parts = []
        for k in range(K):
            mask = part_masks[:, k:k+1]
            part = x_for_seg * mask
            feat = self.clip_model.encode_image(part)  # (B,D)
            img_feats_parts.append(feat)
        img_feats_parts = torch.stack(img_feats_parts, dim=1)  # (B,K,D)

        # 4) text embedding
        text_feats_full = self.clip_model.encode_text(texts)   # (B,D)
        text_feats_parts = text_feats_full.unsqueeze(1).repeat(1, K, 1)

        # 5) Local similarity (simple 1-1 matching by max)
        sim_local = torch.nn.functional.cosine_similarity(
            text_feats_parts, img_feats_parts, dim=-1
        )  # (B,K)
        max_sim, _ = sim_local.max(dim=-1)                     # (B,)

        # 6) global similarity
        img_feat_global = self.clip_model.encode_image(x_for_seg)
        sim_global = torch.nn.functional.cosine_similarity(
            text_feats_full, img_feat_global, dim=-1
        )

        loss_local = -max_sim.mean()
        loss_global = -sim_global.mean()

        return self.lambda_cons * loss_local + self.lambda_global * loss_global