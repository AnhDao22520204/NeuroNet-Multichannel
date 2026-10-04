from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception:  # optional dependency
    Mamba = None


class TemporalContextBase(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_dim: int, embed_dim: int, unfreeze_last_encoder_blocks: int = 1):
        super().__init__()
        self.backbone = backbone
        self._configure_backbone(unfreeze_last_encoder_blocks)
        self.embed = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def _configure_backbone(self, n_last_blocks: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = getattr(self.backbone, "encoder_block", None)
        if blocks is not None and n_last_blocks > 0:
            for block in list(blocks)[-int(n_last_blocks):]:
                for p in block.parameters():
                    p.requires_grad = True
        norm = getattr(self.backbone, "encoder_norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True

    def encode_sequence(self, x):
        # x: (B,L,C,T)
        features = []
        for t in range(x.shape[1]):
            feat = self.backbone(x[:, t])
            feat = self.embed(feat)
            features.append(feat)
        return torch.stack(features, dim=1)


class LSTMTCM(TemporalContextBase):
    def __init__(self, backbone, backbone_dim, embed_dim=256, unfreeze_last_encoder_blocks=1):
        super().__init__(backbone, backbone_dim, embed_dim, unfreeze_last_encoder_blocks)
        self.temporal = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.encode_sequence(x)
        x, _ = self.temporal(x)
        return self.classifier(x)


class MHATCM(TemporalContextBase):
    def __init__(self, backbone, backbone_dim, embed_dim=256, unfreeze_last_encoder_blocks=1):
        super().__init__(backbone, backbone_dim, embed_dim, unfreeze_last_encoder_blocks)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        return self.classifier(self.temporal(self.encode_sequence(x)))


class LSTMMHATCM(TemporalContextBase):
    def __init__(self, backbone, backbone_dim, embed_dim=256, unfreeze_last_encoder_blocks=1):
        super().__init__(backbone, backbone_dim, embed_dim, unfreeze_last_encoder_blocks)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.mha = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x, _ = self.lstm(self.encode_sequence(x))
        return self.classifier(self.mha(x))


class MambaTCM(TemporalContextBase):
    def __init__(self, backbone, backbone_dim, embed_dim=256, unfreeze_last_encoder_blocks=1):
        super().__init__(backbone, backbone_dim, embed_dim, unfreeze_last_encoder_blocks)
        if Mamba is None:
            raise ImportError("mamba_ssm is optional; install requirements-mamba.txt")
        self.temporal = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        return self.classifier(self.temporal(self.encode_sequence(x)))


def build_tcm(name: str, backbone, backbone_dim: int, embed_dim: int, unfreeze_last_encoder_blocks: int = 1):
    name = name.lower()
    classes = {"lstm": LSTMTCM, "mha": MHATCM, "lstm_mha": LSTMMHATCM, "mamba": MambaTCM}
    if name not in classes:
        raise ValueError(f"Unknown TCM {name}")
    return classes[name](backbone, backbone_dim, embed_dim, unfreeze_last_encoder_blocks)
