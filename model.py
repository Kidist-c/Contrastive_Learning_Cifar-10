"""

--------
Encoder f(x) -> h   and   Projection head g(h) -> z

The encoder is what you keep after training (used for downstream tasks).
The projection head only exists to give the contrastive loss "room to work"
without forcing the main representation h to become fully invariant to
every augmentation.

"CNN fits image data because:

Spatial locality — convolution filters look at local pixel neighborhoods (edges, curves), which is how image meaning is actually encoded. 
                   An MLP flattens the image first and loses that 2D structure.
Translation invariance — your augmentations include random crops/flips, so the same object can shift position across a positive pair's two views. CNNs handle this naturally (a filter fires on a pattern wherever it appears); MLPs don't, and 
                         invariance is exactly what contrastive learning is trying to teach.
Parameter efficiency — weight sharing across spatial positions means far fewer parameters than a fully-connected layer over every pixel.
Hierarchical features — stacked conv layers build pixels → edges → shapes → patterns → embedding, matching how visual structure actually composes.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Small CNN encoder. Works for both MNIST (1 channel) and CIFAR-10 (3 channels)."""

    def __init__(self, in_channels=1, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global average pool -> [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.flatten(1)
        h = self.fc(feat)
        return h  # NOTE: intentionally NOT normalized here; normalization happens on z (or h at eval time)


class ProjectionHead(nn.Module):
    """MLP projection head: h -> z. Standard SimCLR-style 2-layer MLP."""

    def __init__(self, embedding_dim=128, projection_dim=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, h):
        return self.net(h)


class ContrastiveModel(nn.Module):
    """
    Wraps encoder + (optional) projection head.


    """

    def __init__(self, in_channels=1, embedding_dim=128, projection_dim=64,
                 use_projection_head=True):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, embedding_dim=embedding_dim)
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projector = ProjectionHead(embedding_dim=embedding_dim, projection_dim=projection_dim)
        else:
            self.projector = None

    def forward(self, x):
        h = self.encoder(x)
        if self.use_projection_head:
            z = self.projector(h)
        else:
            z = h  # loss operates directly on the encoder representation
        return h, z