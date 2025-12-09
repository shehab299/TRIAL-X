import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from tashkeel.constants import char2id, pos2id, DATA_DIR
from tashkeel.utils import load_data, compute_class_weights, load_checkpoint
from tashkeel.preprocess import build_vocab

from tashkeel.models.abandah_2020.model import AbandahModel
from tashkeel.models.abandah_2020.dataset import AbandahDataset, collate_fn
from tashkeel.models.abandah_2020.trainer import train
from tashkeel.models.abandah_2020.evaluate import evaluate

def main(
    test_name,
    epochs,
    batch_size,
    lr,
    Ts,
    Tw,
    stride,
    device,
    checkpoint=None,
):
    print("Loading data...")
    train_sentences = load_data(os.path.join(DATA_DIR, "new_train.txt"))
    val_sentences = load_data(os.path.join(DATA_DIR, "new_val.txt"))

    print("Building vocab...")
    word2id = build_vocab(train_sentences)

    if device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    class_weights = compute_class_weights(train_sentences).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Initializing model...")
    model = AbandahModel(
        char_vocab_size=len(char2id),
        word_vocab_size=len(word2id),
        pos_feature_dim=len(pos2id),
        char_embed_dim=64,
        char_hidden_size=256,
        char_num_layers=2,
        word_embed_dim=300,
        word_hidden_size=256,
        word_num_layers=2,
        num_classes=15, # Assuming 15 classes from constants
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)
    
    if checkpoint:
        start_epoch = load_checkpoint(checkpoint, model, optimizer, device)
    else:
        start_epoch = 0

    print("Creating datasets...")
    train_dataset = AbandahDataset(
        sentences=train_sentences,
        char2id=char2id,
        word2id=word2id,
        pos2id=pos2id,
        pad_label=-100,
        Ts=Ts,
        Tw=Tw,
        stride=stride,
    )

    val_dataset = AbandahDataset(
        sentences=val_sentences,
        char2id=char2id,
        word2id=word2id,
        pos2id=pos2id,
        pad_label=-100,
        Ts=Ts,
        Tw=Tw,
        stride=stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    save_path = f"checkpoints/{test_name}_best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs,
        evaluate,
        save_path,
        monitor="DER_ALL",
        start_epoch=start_epoch,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Abandah (2020) Diacritization Model")

    parser.add_argument(
        "--test_name", type=str, default="abandah_run", help="Name of the test"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--Ts", type=int, default=10, help="Sentence length")
    parser.add_argument("--Tw", type=int, default=13, help="Word length")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )

    args = parser.parse_args()

    main(
        test_name=args.test_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        Ts=args.Ts,
        Tw=args.Tw,
        stride=args.stride,
        device=args.device,
        checkpoint=args.checkpoint,
    )
