import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from tqdm.notebook import tqdm
import utils
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from enum import Enum

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

def get_metrics(num_classes : int) -> torchmetrics.MetricTracker:
    return torchmetrics.MetricTracker(torchmetrics.MetricCollection({
        Metrics.ACCURACY.value: MulticlassAccuracy(num_classes, average="micro"),
        Metrics.PRECISION.value: MulticlassPrecision(num_classes,average= "macro"),
        Metrics.RECALL.value: MulticlassRecall(num_classes=num_classes,average="macro"),
        Metrics.LOSS.value: CrossEntropyLoss()})
    )

def eval(m: nn.Module, dl: DataLoader, num_classes:int, device: torch.device, tracker:torchmetrics.MetricTracker | None = None):
    if tracker is None:
        tracker = get_metrics(num_classes)

    tracker.increment()
    with torch.inference_mode():
        m.to(device)
        m.eval()
        pbar = tqdm(dl, desc="Evaluating")
        for batch, ls in pbar:
            batch_d = batch.to(device)
            logits_d = m(batch_d)
            tracker(logits_d.cpu(), ls)
            del batch_d
            del logits_d
        metrics = tracker.compute()
    return metrics

def eval_show(m: nn.Module, dataset, n:int=8, page:int=0):
    m.cpu()
    m.eval()
    imgs = []
    ls = []
    preds = []
    with torch.inference_mode():
        for i in range(n):
            img, l = dataset[n*page + i]
            batch = img.unsqueeze(0)
            logits = m(batch)
            imgs.append(img)
            ls.append(l)
            preds.append(torch.argmax(logits, dim=1).item())

    cols = 8
    rows = math.ceil(n/cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    for i, (ax, img, l, pred) in enumerate(zip(axs.flatten(), imgs, ls, preds)):
        ax.imshow(utils.whitened_to_PIL(img))
        ax.axis("off")
        color = "green" if l == pred else "red"
        ax.set_title(dataset.classes[l] + "\npred: " + dataset.classes[pred], fontsize=8, color=color)
    fig.set_figwidth(15)

class Metrics(Enum):
    ACCURACY = "Accuracy"
    LOSS  = "Loss"
    PRECISION = "Precision"
    RECALL = "Recall"

def plot_metrics(train_metrics: dict, test_metrics: dict):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(train_metrics[Metrics.ACCURACY.value], label="train accuracy")
    axs[0].plot(test_metrics[Metrics.ACCURACY.value], label="test accuracy")
    axs[0].legend()
    axs[0].set_title(Metrics.ACCURACY.value)
    axs[1].plot(train_metrics[Metrics.LOSS.value], label="train loss")
    axs[1].plot(test_metrics[Metrics.LOSS.value], label="test loss")
    axs[1].legend()
    axs[1].set_title(Metrics.LOSS.value)

    fig.set_figwidth(15)
