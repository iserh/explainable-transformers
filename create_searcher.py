import time
from pathlib import Path

import h5py
import numpy as np
import scann
import torch
from tqdm import tqdm

data_dir = Path("~/data/AmazonCat-13K").expanduser()

f = h5py.File(data_dir / "amazoncat_encoded.hdf5", "r")

train_embed = f["train"]["embedding"]
train_index = f["train"]["index"]
train_label = f["train"]["label"]

test_embed = f["test"]["embedding"]
test_index = f["test"]["index"]
test_label = f["test"]["label"]

print(train_embed.shape)
print(test_embed.shape)

searcher = (
    scann.scann_ops_pybind.builder(train_embed, 10, "dot_product")
    .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
    .score_ah(2, anisotropic_quantization_threshold=0.2)
    .reorder(100)
    .build()
)

scann_path = data_dir / "scann"
scann_path.mkdir(exist_ok=True)
searcher.serialize(str(scann_path))

scann_path = data_dir / "scann"
scann_path.mkdir(exist_ok=True)
searcher.serialize(str(scann_path))
