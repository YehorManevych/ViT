import numpy as np
import torch
import torchmetrics
import torchvision
import torch.nn as nn

# ImageNet statistics used for whitening
mean= np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def whitened_to_PIL(img):
    return torchvision.transforms.ToPILImage()((img * std.reshape(3,1,1))+mean.reshape(3,1,1))

class CrossEntropyLoss(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.lossf = nn.CrossEntropyLoss()
        self.add_state("loss", default=torch.tensor(0.))
        self.add_state("n", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.loss += self.lossf(preds, target)
        self.n = self.n + target.numel()

    def compute(self):
        return self.loss / self.n

def metrics_to_str(metrics: dict):
    return ', '.join((f"{k} = {v.item():.3f}" for k, v in metrics.items()))
