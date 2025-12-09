import torch
import pickle
import os
from sklearn.utils import compute_class_weight
import re
from tashkeel.preprocess import extract_letters_and_diacritics
import numpy as np

def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences




def compute_class_weights(sentences):
    
    all_labels = []

    for sentence in sentences:
        clean = re.sub(r"[^ء-ي\u064B-\u0652\s]", "", sentence)
        tokens = clean.split()

        for word in tokens:
            letters, labels = extract_letters_and_diacritics(word)
            all_labels.extend(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels,
    )

    return torch.tensor(class_weights, dtype=torch.float)

def load_checkpoint(checkpoint, model, optimizer, device):

    if checkpoint is not None:
     
        if torch.cuda.is_available():
            ckpt = torch.load(checkpoint)
        else:
            ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1  

        print(f"Loaded checkpoint from '{checkpoint}'")
        print(f"Resuming from epoch {start_epoch}")
        if "best_metric" in ckpt:
            print(f"Previous best metric: {ckpt['best_metric']:.4f}")

        return start_epoch