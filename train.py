import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import utils
import eval
import math

def train(m: nn.Module, epochs:int, train_dl: DataLoader, test_dl: DataLoader, device:torch.device, num_classes:int, opt:torch.optim.Optimizer, lossf:nn.Module):
    eval_n_times = 5
    eval_every_n_batches = math.ceil(len(train_dl)/eval_n_times)
    m = m.to(device)
    args = ("multiclass", 0.5, num_classes)
    train_metricsf = torchmetrics.MetricCollection(
        torchmetrics.Accuracy(*args), 
        torchmetrics.Precision(*args),
        torchmetrics.Recall(*args),
        utils.CrossEntropyLoss()).to(device)
    train_metrics = []
    test_metrics = []
    for epoch in range(epochs):
        m.train()
        print(f"Epoch {epoch}:")
        for i, (batch, ls) in enumerate(train_dl):
            batch, ls = batch.to(device), ls.to(device)
            print(f"\tBatch {i}")
            logits = m(batch)

            loss = lossf(logits, ls)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.inference_mode():
                train_metricsf(logits.cpu(), ls.cpu())

                should_print = i % eval_every_n_batches == 0 or i == len(train_dl)-1
                if should_print:
                    train_m = train_metricsf.compute()
                    print(f"\n\tTrain: {utils.metrics_to_str(train_m)}")
                    train_metrics.append(train_m)
                    train_metricsf.reset()

                    test_m = eval.eval(m, test_dl, num_classes, device)
                    print(f"\n\tTest: {utils.metrics_to_str(test_m)}")
                    test_metrics.append(test_m)
    return train_metrics, test_metrics