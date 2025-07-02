import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
'''
class DepthwisePointwise3D(nn.Module):
    """Standard Depthwise -> Pointwise 3D block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwisePointwise3D, self).__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualDepthwisePointwise3D(nn.Module):
    """Depthwise–Pointwise block with residual skip if dimensions match."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualDepthwisePointwise3D, self).__init__()

        self.block = DepthwisePointwise3D(in_channels, out_channels, kernel_size, padding)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.block(x)
        if self.shortcut:
            x = self.shortcut(x)
        return torch.relu(out + x)


class Custom_3D_CNN(nn.Module):
    """3D CNN with Residual Depthwise-Pointwise Layers."""
    def __init__(self, num_classes):
        super(Custom_3D_CNN, self).__init__()

        self.features = nn.Sequential(
            ResidualDepthwisePointwise3D(1, 32),
            nn.MaxPool3d((1, 2, 2)),

            ResidualDepthwisePointwise3D(32, 64),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            ResidualDepthwisePointwise3D(64, 128),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            ResidualDepthwisePointwise3D(128, 64),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            ResidualDepthwisePointwise3D(64, 64),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
-------------------------------------------------------------------
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight

    def forward(self, x):
        B, C, D, H, W = x.shape
        query = self.query(x).view(B, -1, D * H * W)   # (B, C//8, N)
        key   = self.key(x).view(B, -1, D * H * W)     # (B, C//8, N)
        value = self.value(x).view(B, -1, D * H * W)   # (B, C, N)

        attn = torch.bmm(query.permute(0, 2, 1), key)  # (B, N, N)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(value, attn.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, D, H, W)

        return self.gamma * out + x
    

class Residual3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout=0.2, use_attention=False):
        super(Residual3DBlock, self).__init__()
        stride = (1, 2, 2) if downsample else (1, 1, 1)

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )

        # If dimensions don't match, project input to match output
        self.skip_connection = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention3D(out_channels)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv_block(x)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out


class Custom_3D_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Custom_3D_CNN, self).__init__()

        self.features = nn.Sequential(
            Residual3DBlock(1, 32, downsample=True),
            Residual3DBlock(32, 64, downsample=True, dropout=0.3),
            Residual3DBlock(64, 128, downsample=True, dropout=0.3),
            Residual3DBlock(128, 64, downsample=True, dropout=0.3, use_attention=True),
            Residual3DBlock(64, 64, downsample=True, dropout=0.3, use_attention=True),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
'''
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class GlobalAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        weights = self.attn(x)                    # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        weighted = x * weights                    # (B, T, D)
        return weighted.sum(dim=1)                # (B, D)

class Custom_3D_CNN(nn.Module):
    def __init__(self, num_classes, embed_dim=64, num_heads=8, num_layers=6):
        super(Custom_3D_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.3),

            nn.Conv3d(64, embed_dim, kernel_size=1)  # projection
        )

        self.pool = nn.AdaptiveAvgPool3d((1, None, None))  # preserve spatial grid

        self.pos_enc = PositionalEncoding(embed_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout=0.3) for _ in range(num_layers)
        ])

        self.attn_pool = GlobalAttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, 10, 64, 65)
        x = self.cnn(x)                        # (B, embed_dim, D, H, W)
        x = self.pool(x)  # (B, embed_dim, 1, H', W')
        x = x.squeeze(2)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, T, embed_dim)

        x = self.pos_enc(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.attn_pool(x)  # (B, embed_dim)
        return self.classifier(x)



class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path='checkpoint.pth'):
        """Stop training when validation loss doesn’t improve after a given patience."""
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        """Check if the model has improved and save the best one."""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        """Save model when a new best loss is found."""
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: class weight tensor of shape [num_classes] or scalar
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
