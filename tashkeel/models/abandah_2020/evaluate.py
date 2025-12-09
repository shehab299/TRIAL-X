import torch
from tashkeel.metrics import compute_DER

def evaluate(model, loader, device):
    der_ce_total = []
    der_cw_total = []
    der_all_total = []

    with torch.no_grad():
        for words, chars, diacs, pos, gender, number, person, lens in loader:
            words = words.to(device)
            chars = chars.to(device)
            diacs = diacs.to(device)
            pos = pos.to(device)
            gender = gender.to(device)
            number = number.to(device)
            person = person.to(device)
            lens = lens.to(device)
            
            outputs = model(words, chars, pos, gender, number, person, lens)

            der_ce, der_cw, der_all = compute_DER(outputs, diacs, chars)

            der_ce_total.append(der_ce)
            der_cw_total.append(der_cw)
            der_all_total.append(der_all)

    der_ce = sum(der_ce_total) / len(der_ce_total)
    der_cw = sum(der_cw_total) / len(der_cw_total)
    der_all = sum(der_all_total) / len(der_all_total)

    return der_ce, der_cw, der_all
