import torch


def compute_DER(outputs, labels, chars):
    """
    Inputs:
    ---
    outputs: model outputs (B, Ts, Tw, C)
    labels: True diacritics  (B, Ts, Tw)
    chars:   (B, Ts, Tw)


    Outputs:
    ---
    DER_CE: diacritic error rate for character ending
    DER_CW: diacritic error rate for core word
    DER_ALL: diacritic error rate for all characters
    """

    # predictions
    preds = outputs.argmax(dim=-1)

    # masks
    char_mask = chars != 0

    shifted = torch.zeros_like(char_mask)
    shifted[..., :-1] = char_mask[..., 1:]

    last_letter_mask = char_mask & (~shifted)
    within_word_mask = char_mask & (~last_letter_mask)

    # DER for last letter only
    ce_p = preds[last_letter_mask]
    ce_g = labels[last_letter_mask]
    der_ce = (ce_p != ce_g).float().mean().item()

    # DER for all other characters
    cw_p = preds[within_word_mask]
    cw_g = labels[within_word_mask]
    der_cw = (cw_p != cw_g).float().mean().item()

    # DER for all characters
    all_p = preds[char_mask]
    all_g = labels[char_mask]
    der_all = (all_p != all_g).float().mean().item()

    return der_ce, der_cw, der_all
