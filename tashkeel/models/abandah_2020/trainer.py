import torch
import os
from tqdm import tqdm

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    evaluate_fn,
    save_path="best_model.pt",
    monitor="DER_ALL",
    start_epoch=0,
):
    best_metric = float("inf")

    for epoch in range(start_epoch, start_epoch + epochs):

        model.train()
        total_loss = 0.0

        for words, chars, labels, pos, gender, number, person, lens in tqdm(train_loader):
            words = words.to(device)
            chars = chars.to(device)
            labels = labels.to(device)
            pos = pos.to(device)
            gender = gender.to(device)
            number = number.to(device)
            person = person.to(device)
            lens = lens.to(device)

            outputs = model(words, chars, pos, gender, number, person, lens)  # (B, Ts, Tw, C)
            logits = outputs.reshape(-1, outputs.size(-1))  # (B*Ts*Tw, C)
            labels_flat = labels.reshape(-1)

            loss = criterion(logits, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)

        # Set model to eval mode before validation
        model.eval()
        DER_CE_Train, DER_CW_Train, DER_ALL_Train = evaluate_fn(
            model, train_loader, device
        )
        DER_CE, DER_CW, DER_ALL = evaluate_fn(model, val_loader, device)

        print(
            f"Epoch {epoch}: train_loss={epoch_loss:.4f}, DER_CE={DER_CE_Train:.4f}, DER_CW={DER_CW_Train:.4f}, DER_ALL={DER_ALL_Train:.4f}"
        )
        print(
            f"Epoch {epoch}: train_loss={epoch_loss:.4f}, DER_CE={DER_CE:.4f}, DER_CW={DER_CW:.4f}, DER_ALL={DER_ALL:.4f}"
        )

        # Determine which metric to track
        current_metric = DER_CE if monitor == "DER_CE" else DER_CW
        if monitor == "DER_ALL":
            current_metric = DER_ALL

        # Save best model
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": best_metric,
                },
                save_path,
            )
            print(
                f">> Saved best model at epoch {epoch} with {monitor}={best_metric:.4f}"
            )

    print(f"Training complete. Best {monitor}: {best_metric:.4f}")
