import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import eval
import math
import utils
from tqdm.notebook import tqdm

def train(m: nn.Module, epochs:int, train_dl: DataLoader, test_dl: DataLoader, device:torch.device, num_classes:int, opt:torch.optim.Optimizer, lossf:nn.Module, evals_per_epoch:int=5, checkpoints:int =0):
    batches = len(train_dl)
    # running window for the train metrics
    metric_window = math.ceil(batches/evals_per_epoch)
    m = m.to(device)
    train_tracker = eval.get_metrics(num_classes)
    test_tracker = eval.get_metrics(num_classes)
    epochs_per_checkpoint = math.ceil(epochs/checkpoints)
    ebar = tqdm(range(epochs), desc="Epochs")
    for epoch in ebar:
        ebar.set_description(f"Epoch {epoch}")
        ebar.write(f"Epoch {epoch}")
        bbar = tqdm(enumerate(train_dl), total=batches, desc="Batches")
        for i, (batch, ls) in bbar:
            bbar.set_description(f"Batch {i}")

            m.train()
            batch_d, ls_d = batch.to(device), ls.to(device)
            logits_d = m(batch_d)

            loss_d = lossf(logits_d, ls_d)
            opt.zero_grad()
            loss_d.backward()
            opt.step()

            with torch.inference_mode():
                window_started = i % metric_window == 0
                window_ended = (i+1) % metric_window == 0 or i == batches - 1
                if(window_started):
                    train_tracker.increment()

                train_tracker(logits_d.cpu(), ls)

                if window_ended:
                    m.eval()
                    bbar.write(f"Batch {i}:")
                    bbar.write(f"Train: {eval.metrics_to_str(train_tracker.compute())}")

                    bbar.write(f"Testing:")
                    test_m = eval.eval(m, test_dl, num_classes, device, test_tracker)
                    bbar.write(f"Test: {eval.metrics_to_str(test_m)}\n")

                    if((epoch+1) % epochs_per_checkpoint == 0):
                        utils.save_model(m, test_m["Accuracy"].item(), f"e{epoch}")


            del batch_d
            del ls_d
            del loss_d
            del logits_d
        ebar.write("\n")

    return train_tracker.compute_all(), test_tracker.compute_all()