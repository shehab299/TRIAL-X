# ==========================================
import torch

# ==========================================
from tashkeel.metrics import compute_DER

# ==========================================


def evaluate(model, loader, device):
    """
    Standard evaluation without majority voting.
    Ignores metadata and evaluates each segment independently.
    """
    der_ce_total = []
    der_cw_total = []
    der_all_total = []

    with torch.no_grad():
        for word_embeddings, chars, diacs in loader:

            word_embeddings = word_embeddings.to(device)
            chars = chars.to(device)
            diacs = diacs.to(device)

            outputs = model(word_embeddings, chars)

            der_ce, der_cw, der_all = compute_DER(outputs, diacs, chars)

            der_ce_total.append(der_ce)
            der_cw_total.append(der_cw)
            der_all_total.append(der_all)

    der_ce = sum(der_ce_total) / len(der_ce_total)
    der_cw = sum(der_cw_total) / len(der_cw_total)
    der_all = sum(der_all_total) / len(der_all_total)

    return der_ce, der_cw, der_all
