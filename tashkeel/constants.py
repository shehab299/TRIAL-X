import os
import pickle
from farasa import FarasaPOSTagger


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


posTagger = FarasaPOSTagger(interactive=True)

BASE_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
UTILS_DIR = os.path.join(DATA_DIR, "utils")


arabic_letters = load_pickle(os.path.join(UTILS_DIR, "arabic_letters.pickle"))
diacritics = load_pickle(os.path.join(UTILS_DIR, "diacritics.pickle"))
diacritic2id = load_pickle(os.path.join(UTILS_DIR, "diacritic2id.pickle"))
pos2id = load_pickle(os.path.join(UTILS_DIR, "pos2id.pickle"))
id2diacritic = {v: k for k, v in diacritic2id.items()}

CHAR_PAD_TOKEN = "P"
chars = [CHAR_PAD_TOKEN] + sorted(list(arabic_letters))


char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for c, i in char2id.items()}

id2diactic_name = {
    -100: "PAD",
    0: "fatha",
    1: "tanween_fatha",
    2: "damma",
    3: "tanween_damma",
    4: "kasra",
    5: "tanween_kasra",
    6: "sukun",
    7: "shadda",
    8: "shadda_fatha",
    9: "shadda_tanween_fatha",
    10: "shadda_damma",
    11: "shadda_tanween_damma",
    12: "shadda_kasra",
    13: "shadda_tanween_kasra",
    14: "no_diacritic",
}
