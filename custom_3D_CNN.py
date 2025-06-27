import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
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


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
