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
    best_metric = float("inf")  # since DER (error rate) should be minimized

    for epoch in range(start_epoch, start_epoch + epochs):

        model.train()
        total_loss = 0.0

        for words, chars, labels, pos in tqdm(train_loader):
            # metadata not needed during training, only for evaluation
            words = words.to(device)
            chars = chars.to(device)
            labels = labels.to(device)
            pos = pos.to(device)

            outputs = model(words, chars, pos)  # (B, Ts, Tw, C)
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


def evaluate_crf(model, loader, device):
    der_ce_total = []
    der_cw_total = []
    der_all_total = []

    with torch.no_grad():
        for words, chars, labels, pos in loader:
            words = words.to(device)
            chars = chars.to(device)
            labels = labels.to(device)
            pos = pos.to(device)

            # Create mask for CRF (ignore padding)
            mask = labels != -100

            # Forward pass with CRF (inference)
            # returns predictions (B, Ts, Tw)
            preds = model(words, chars, pos, mask=mask)

            # --- Compute DER manually since compute_DER expects logits ---
            
            # masks
            char_mask = chars != 0 # Assuming 0 is PAD char
            
            shifted = torch.zeros_like(char_mask)
            shifted[..., :-1] = char_mask[..., 1:]
            
            last_letter_mask = char_mask & (~shifted)
            within_word_mask = char_mask & (~last_letter_mask)
            
            # Filter out padding from labels for comparison
            # Note: preds from CRF might already handle masking, but we need to align with labels
            
            # DER for last letter only
            ce_p = preds[last_letter_mask]
            ce_g = labels[last_letter_mask]
            # Ignore -100 in labels if any (though char_mask should handle it)
            valid_ce = ce_g != -100
            if valid_ce.sum() > 0:
                der_ce = (ce_p[valid_ce] != ce_g[valid_ce]).float().mean().item()
                der_ce_total.append(der_ce)

            # DER for all other characters
            cw_p = preds[within_word_mask]
            cw_g = labels[within_word_mask]
            valid_cw = cw_g != -100
            if valid_cw.sum() > 0:
                der_cw = (cw_p[valid_cw] != cw_g[valid_cw]).float().mean().item()
                der_cw_total.append(der_cw)

            # DER for all characters
            all_p = preds[char_mask]
            all_g = labels[char_mask]
            valid_all = all_g != -100
            if valid_all.sum() > 0:
                der_all = (all_p[valid_all] != all_g[valid_all]).float().mean().item()
                der_all_total.append(der_all)

    der_ce = sum(der_ce_total) / len(der_ce_total) if der_ce_total else 0.0
    der_cw = sum(der_cw_total) / len(der_cw_total) if der_cw_total else 0.0
    der_all = sum(der_all_total) / len(der_all_total) if der_all_total else 0.0

    return der_ce, der_cw, der_all


def train_crf(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    save_path="best_model_crf.pt",
    monitor="DER_ALL",
    start_epoch=0,
):
    best_metric = float("inf")

    for epoch in range(start_epoch, start_epoch + epochs):

        model.train()
        total_loss = 0.0

        for words, chars, labels, pos in tqdm(train_loader):
            words = words.to(device)
            chars = chars.to(device)
            labels = labels.to(device)
            pos = pos.to(device)

            # Create mask for CRF (ignore padding)
            # labels are -100 for padding
            mask = labels != -100

            # Forward pass with targets returns loss
            loss = model(words, chars, pos, targets=labels, mask=mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)

        # Set model to eval mode before validation
        model.eval()
        
        # Use internal evaluate_crf
        DER_CE_Train, DER_CW_Train, DER_ALL_Train = evaluate_crf(
            model, train_loader, device
        )
        DER_CE, DER_CW, DER_ALL = evaluate_crf(model, val_loader, device)

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
