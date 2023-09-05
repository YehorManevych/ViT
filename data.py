import torchvision
import torch.nn as nn
from pathlib import Path
import json
import shutil
import os
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
from typing import Tuple
import math
import requests
import tarfile
import numpy as np

DATASETS_DIR = Path("datasets")
IMAGNETTE2_PATH = DATASETS_DIR / "imagenette2"
IMAGNETTE2_FORMAT_PATH = DATASETS_DIR / "imagenette2_format"
IMAGNETTE2_ARCHIVE_PATH = DATASETS_DIR / "imagenette2.tgz"
CIFAR_PATH = DATASETS_DIR / "CIFAR-10"

def get_child_dirs(root:Path):
    return (d for d in os.scandir(root) if not d.name.startswith("."))

def download_imagenette2() -> Path:
    imagenette2_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

    if DATASETS_DIR.exists() and IMAGNETTE2_PATH.exists():
        print(f"Imagenette2 is already downloaded! {IMAGNETTE2_PATH.absolute()}")
        return IMAGNETTE2_PATH

    if not DATASETS_DIR.exists():
        print(f"Creating {DATASETS_DIR}")
        DATASETS_DIR.mkdir()
    
    if not IMAGNETTE2_PATH.exists():
        print(f"Creating {DATASETS_DIR}")
        IMAGNETTE2_PATH.mkdir()

    print(f"Downloading Imagenette2 from {imagenette2_url}")
    with requests.get(imagenette2_url, stream=True) as resp:
        chunk_size =  1024 * 1024 # 1MB
        file_size = int(resp.headers["Content-Length"])
        print(f"File size is {file_size} bytes")
        total = math.ceil(file_size/chunk_size)
        pbar = tqdm(resp.iter_content(chunk_size=chunk_size), total=total, desc="Downloading")
        for chunk in pbar:
            with open(IMAGNETTE2_ARCHIVE_PATH, 'ab') as fd:
                fd.write(chunk)
    pbar.write(f"Archive is downloaded! {IMAGNETTE2_ARCHIVE_PATH.absolute()}")
    pbar.write("Extracting the archive")
    with tarfile.open(IMAGNETTE2_ARCHIVE_PATH) as archive:
        archive.extractall(DATASETS_DIR)
    pbar.write(f"Archive is extracted to {IMAGNETTE2_PATH.absolute()}")
    return IMAGNETTE2_PATH


def format_imagenette2(classes_json : Path)\
    -> Tuple[Path, list[str]]:
    assert IMAGNETTE2_PATH.exists(), f"Directory {IMAGNETTE2_PATH.absolute()} does not exist"

    class_by_id = {}
    with open(classes_json) as f:
        cbi_json = json.load(f)
    for _, cbi in cbi_json.items():
        class_by_id[cbi[0]]= cbi[1]
    classes = list(class_by_id.values())

    output_path = IMAGNETTE2_FORMAT_PATH
    if output_path.exists():
        print(f"Imagenette2 dataset is already formatted and exists at {output_path.absolute()}")
        return output_path, classes

    print("Formatting Imagenette2 dataset")
    train_path =  output_path / "train"
    test_path =  output_path / "test"
    shutil.copytree(IMAGNETTE2_PATH, output_path)

    train_dirs = get_child_dirs(train_path)
    print("Renaming train")
    for d in train_dirs:
        os.rename(d.path, train_path / class_by_id[d.name])
    os.rename(output_path / "val", test_path)

    test_dirs = get_child_dirs(test_path)
    print("Renaming test")
    for d in test_dirs:
        os.rename(d.path, test_path / class_by_id[d.name])

    print(f"Imagenette2 is ready! {output_path.absolute()}")
    return output_path, classes

class Imagenette2(ImageFolder):
        def __init__(self, root:Path, classes:list[str], transform):
            self.root = root
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

def load_imagenette2(classes:list[str], batch:int, transforms:nn.Module) \
    -> Tuple[ImageFolder, ImageFolder, DataLoader, DataLoader]:
    train_ds = Imagenette2(IMAGNETTE2_FORMAT_PATH / "train", classes, transforms)
    test_ds = Imagenette2(IMAGNETTE2_FORMAT_PATH / "test", classes, transforms)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch, shuffle=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch, shuffle=True)
    return train_ds, test_ds, train_dl, test_dl

def load_CIFAR(train:bool, batch_size:int, max_batches_n:int | None = None, shuffle:bool=True, transforms:nn.Module | None = None, seed:int | None = None) \
    -> Tuple[torchvision.datasets.CIFAR10, DataLoader]:
    ds = torchvision.datasets.CIFAR10(str(CIFAR_PATH), download=True, train=train, transform=transforms)

    subset = ds
    if max_batches_n is not None:
        length = batch_size * max_batches_n
        length = length if length <= len(ds) else len(ds)
        
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(length)
        np.random.shuffle(indices)
        subset = Subset(ds, indices.tolist())

    dl = DataLoader(dataset=subset, batch_size=batch_size, shuffle=shuffle)
    print(f"Created {len(dl)} {'train' if train else 'test'} batches of size {batch_size}")
    return ds, dl