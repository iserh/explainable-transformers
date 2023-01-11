from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from helpers import AmazonCat13K, ColBERT


def prepare(dataset, model, tokenizer, output_file: h5py.File, split: str, batch_size: int) -> None:
    print(f"\n\nSPLIT: {split}\n")

    # create dataloader for this dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    labels = torch.nn.utils.rnn.pad_sequence(dataset.df["target_ind"].apply(torch.LongTensor), batch_first=True)
    ds_label = output_file.create_dataset(f"{split}/label", data=labels)

    ds_embedding = output_file.create_dataset(
        f"{split}/embedding", (len(ds_label) * 512, 768), maxshape=(None, 768), dtype="e"
    )
    ds_index = output_file.create_dataset(f"{split}/index", (len(ds_label) * 512,), maxshape=(None,), dtype="i")

    k = 0
    for i, text in enumerate((pbar := tqdm(dataloader))):
        pbar.set_description_str("Encode")

        input_dict = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

        # run model on all devices
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            embedding = model(**input_dict).cpu()

        pbar.set_description_str("Write")

        attention = input_dict["attention_mask"].bool()
        embedding = embedding[attention]
        indices = np.where(attention)[0] + batch_size * i

        ds_embedding[k : k + len(embedding)] = embedding
        ds_index[k : k + len(indices)] = indices

        k += len(embedding)

    ds_embedding.resize(k, axis=0)
    ds_index.resize(k, axis=0)

    print(f"\nSPLIT: {split} - DONE!\n\n")


if __name__ == "__main__":
    DATA_DIR = Path.home() / "data/AmazonCat-13K"
    DEVICE_IDS = list(range(torch.cuda.device_count()))
    print(f"Devices: {DEVICE_IDS}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
    net = torch.nn.DataParallel(model.cuda(DEVICE_IDS[0]), DEVICE_IDS)

    output_file = h5py.File(DATA_DIR / "amazoncat_encoded.hdf5", "w")

    dataset = AmazonCat13K(train=False)
    prepare(
        dataset=dataset,
        model=net,
        tokenizer=tokenizer,
        output_file=output_file,
        split="test",
        batch_size=512,
    )

    dataset = AmazonCat13K(train=True)
    prepare(
        dataset=dataset,
        model=net,
        tokenizer=tokenizer,
        output_file=output_file,
        split="train",
        batch_size=512,
    )
