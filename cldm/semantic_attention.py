# cldm/semantic_attention.py

import torch
from typing import List, Dict, Any

class SemanticBundledAttentionLoss(torch.nn.Module):
    def __init__(self, lambda_bundle: float = 0.5):
        super().__init__()
        self.lambda_bundle = lambda_bundle

    def _jsd(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # p, q: (B, H, W), already normalized as distributions
        p = p.clamp(eps, 1.0)
        q = q.clamp(eps, 1.0)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum(dim=[1, 2])
        kl_qm = (q * (q / m).log()).sum(dim=[1, 2])
        return 0.5 * (kl_pm + kl_qm)

    def forward(self,
                A_adj_list: List[torch.Tensor],
                A_noun_list: List[torch.Tensor]) -> torch.Tensor:
        """
        A_*_list: attention maps (B, H, W) for each adjective/noun pair
        """
        if len(A_adj_list) == 0:
            # no adjective/noun pairs in this sample
            return torch.tensor(0.0, device=A_noun_list[0].device) if A_noun_list else torch.tensor(0.0)

        loss = 0.0
        for A_adj, A_noun in zip(A_adj_list, A_noun_list):
            loss = loss + self._jsd(A_adj, A_noun).mean()
        return self.lambda_bundle * loss


class FashionCompatibilityLoss(torch.nn.Module):
    def __init__(self, encoder, lambda_comp: float = 0.5):
        super().__init__()
        self.encoder = encoder    # fashion-compatibility encoder g(.)
        self.lambda_comp = lambda_comp

    def forward(self, x_img: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        x_img: (B, 3, H, W) in [-1, 1]
        """
        if self.encoder is None:
            return x_img.new_tensor(0.0)

        x_for_comp = (x_img + 1.0) / 2.0
        feat_img = self.encoder.encode_image(x_for_comp)
        feat_txt = self.encoder.encode_text(texts)

        feat_img = torch.nn.functional.normalize(feat_img, dim=-1)
        feat_txt = torch.nn.functional.normalize(feat_txt, dim=-1)

        sim = (feat_img * feat_txt).sum(dim=-1)
        return self.lambda_comp * (-sim.mean())


class CLIPSemanticLoss(torch.nn.Module):
    def __init__(self, clip_model, lambda_perc: float = 0.1):
        super().__init__()
        self.clip_model = clip_model
        self.lambda_perc = lambda_perc

    def forward(self, x_img: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Eq. (19): 1 - cos(f_img(Dec(z_tilde_{t-1})), f_text(T))
        x_img: (B, 3, H, W) in [-1, 1]
        """
        if self.clip_model is None:
            return x_img.new_tensor(0.0)

        x_for_clip = (x_img + 1.0) / 2.0
        img_feat = self.clip_model.encode_image(x_for_clip)
        txt_feat = self.clip_model.encode_text(texts)

        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)

        cos_sim = (img_feat * txt_feat).sum(dim=-1)
        return self.lambda_perc * (1.0 - cos_sim.mean())