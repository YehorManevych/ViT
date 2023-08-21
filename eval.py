import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models.vision_transformer import vit_b_16
from torchvision.datasets import ImageFolder
from pathlib import Path
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from itertools import islice
from plotly.subplots import make_subplots
import plotly.express as px
import math
import sys
from tqdm.notebook import tqdm


def accuracy(pred:torch.Tensor, target:torch.Tensor):
    correct = len((pred==target).nonzero())
    return correct/len(pred)

def errrate(pred:torch.Tensor, target:torch.Tensor):
    wrong = len((pred!=target).nonzero())
    return wrong/len(pred)

def mean(list):
    return sum(list)/len(list)

def eval(m: nn.Module, dl: DataLoader, n_batches:int, lossf: nn.Module):
    m.eval()
    with torch.inference_mode():
        pbar = tqdm(islice(dl, n_batches), leave=True)
        losses = []
        accs = []
        errrs = []
        for batch, ls in pbar:
            logits = m(batch)
            preds = torch.argmax(logits, dim=1)
            loss = lossf(logits, ls).item()
            losses.append(loss)
            acc = accuracy(preds, ls)
            accs.append(acc)
            errr = errrate(preds, ls)
            errrs.append(errr)
            pbar.set_description(f"Loss:{loss:.3f}, acc:{acc*100:.1f}%, err rate:{errr:.3f}")
    pbar.write(f"AVEREGE METRICS:\n\tLoss:{mean(losses):.3f}, acc:{mean(accs)*100:.1f}%, err rate:{mean(errrs):.3f}")



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

    std = np.array([0.229, 0.224, 0.225])
    mean= np.array([0.485, 0.456, 0.406])

    for i, img in enumerate(imgs):
        img_orig = torchvision.transforms.ToPILImage()((img * std.reshape(3,1,1))+mean.reshape(3,1,1))
        row = int(i / cols)
        col = i - row * cols
        fig.add_image(z=img_orig, row = row+1, col = col+1 )

    return fig