import torchvision
import torch.nn as nn
from pathlib import Path
import json
import shutil
import os
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Tuple
import math

def get_child_dirs(root:Path):
    return (d for d in os.scandir(root) if not d.name.startswith("."))

def prepare_imagenette(data_path : Path, classes_json : Path):
    assert data_path.exists(), f"Directory {data_path} does not exist"
    assert classes_json.exists(), f"File {classes_json} does not exist"

    class_by_id = {}
    with open(classes_json) as f:
        cbi_json = json.load(f)
    for _, cbi in cbi_json.items():
        class_by_id[cbi[0]]= cbi[1]
    output_path = Path("data")
    train_path =  output_path / "train"
    test_path =  output_path / "test"
    if not output_path.exists():
        shutil.copytree(data_path, output_path)
        train_dirs = get_child_dirs(train_path)
        for d in train_dirs:
            os.rename(d.path, train_path / class_by_id[d.name])
        os.rename(output_path / "val", test_path)
        test_dirs = get_child_dirs(test_path)
        for d in test_dirs:
            os.rename(d.path, test_path / class_by_id[d.name])
    print("Data is ready!")
    classes = list(class_by_id.values())
    return output_path, classes

class Imagenette2(VisionDataset):
        def __init__(self, root : Path, classes: list[str], transform):
            self.transforms = transform
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.imgs = []
            present_classes = [d.name for d in os.scandir(root) if not d.name.startswith(".")]
            for c in present_classes:
                id = self.class_to_idx[c]
                path = root / c
                for img_path in os.listdir(path):
                    self.imgs.append((path / img_path, id))
            
        def __getitem__(self, index):
            img_path, l =  self.imgs[index]
            img = Image.open(img_path)
            img = img.convert(mode="RGB")
            return self.transforms(img), l
        
        def __len__(self):
            return len(self.imgs)

def load_imagenette(train_data_path: Path, test_data_path: Path, classes: list[str], batch:int, transforms: nn.Module) \
    -> Tuple[VisionDataset, VisionDataset, DataLoader, DataLoader]:
    train_ds = Imagenette2(train_data_path, classes, transforms)
    test_ds = Imagenette2(test_data_path, classes, transforms)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch, shuffle=True)
    return train_ds, test_ds, train_dl, test_dl

def load_cifar(path: str, train:bool, batch_size:int, batch_n:int | None = None, shuffle:bool=True, transforms: nn.Module | None = None) \
    -> Tuple[VisionDataset, DataLoader]:
    ds = torchvision.datasets.CIFAR10(path, download=True, train=train, transform=transforms)

    subset = ds
    if batch_n is not None:
        desired_len = batch_size * batch_n
        skip = math.ceil(len(ds)/desired_len)
        subset = Subset(ds, range(0, len(ds), skip))

    dl = DataLoader(dataset=subset, batch_size=batch_size, shuffle=shuffle)
    print(f"Created {len(dl)} {'train' if train else 'test'} batches of size {batch_size}")
    return ds, dl