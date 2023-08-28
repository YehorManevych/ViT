import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from itertools import islice
from plotly.subplots import make_subplots
import math
from tqdm.notebook import tqdm
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
import utils


def accuracy(pred:torch.Tensor, target:torch.Tensor):
    correct = len((pred==target).nonzero())
    return correct/len(pred)

def errrate(pred:torch.Tensor, target:torch.Tensor):
    wrong = len((pred!=target).nonzero())
    return wrong/len(pred)

def mean(list):
    return sum(list)/len(list)

def eval(m: nn.Module, dl: DataLoader, num_classes:int, device: torch.device):
    average = "macro"
    metricsf = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes),
        MulticlassPrecision(num_classes,average= average),
        MulticlassRecall(num_classes=num_classes,average=average),
        utils.CrossEntropyLoss()])
    with torch.inference_mode():
        m.eval()
        for batch, ls in dl:
            logits = m(batch.to(device))
            metricsf(logits.cpu(), ls)
        metrics = metricsf.compute()
        # print(f"\tTest: {utils.metrics_to_str(metrics)}\n")
    return metrics



def eval_show(m: nn.Module, ds: ImageFolder, n:int=8, page:int=0):
    m.eval()
    imgs = []
    preds = []
    with torch.inference_mode():
        for i in range(n):
            img, l = ds[n*page + i]
            batch = img.unsqueeze(0)
            logits = m(batch)
            imgs.append(img)
            preds.append(torch.argmax(logits, dim=1).item())

    cols = 8
    rows = math.ceil(n/cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=np.array(ds.classes)[preds])

    for i, img in enumerate(imgs):
        row = int(i / cols)
        col = i - row * cols
        fig.add_image(z=utils.whitened_to_PIL(img), row = row+1, col = col+1 )

    return fig