import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.backends.mps
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pathlib import Path 
import os


# ImageNet statistics used for whitening
MEAN= np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

MODELS_DIR = Path("models")

def whitened_to_PIL(img):
    return torchvision.transforms.ToPILImage()((img * STD.reshape(3,1,1))+MEAN.reshape(3,1,1))

def show(ds:ImageFolder, img_i:int):
    img, l = ds[img_i]

    plt.figure(figsize=(2,2))
    plt.imshow(whitened_to_PIL(img))
    plt.axis("off")
    plt.title(ds.classes[l])
    plt.show()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return torch.device(device)

def save_model(m:torch.nn.Module, accuracy:float, desc:str="") -> Path:
    time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    model_name = type(m).__name__
    filename = f"{time}_{model_name}_{round(accuracy*100)}%_{desc}.pth"

    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir()
    path = MODELS_DIR / filename

    torch.save(m.state_dict(), path)
    print(f"Model is saved to {path.absolute()}")
    return path

def load_last_model(m:torch.nn.Module) -> nn.Module:
    model_name = type(m).__name__
    files = sorted(MODELS_DIR.glob(f"*{model_name}*"), key=os.path.getctime)
    assert len(files) != 0, f"No saved {model_name} models found in {MODELS_DIR.absolute()}"

    print(f"Loading model from {files[-1].absolute()}")
    m.load_state_dict(torch.load(files[-1]))
    return m