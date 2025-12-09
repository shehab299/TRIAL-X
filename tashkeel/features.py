from farasa import FarasaPOSTagger
from typing import List
import re
import numpy as np


def extract_pos(sentence: str, pos_tagger: FarasaPOSTagger) -> List[str]:
    """
    Tag a word using the provided FarasaPOSTagger and return only the POS tag(s).
    """
    tagged = pos_tagger.tag(sentence)
    tagged = tagged.replace("S/S", "").replace("E/E", "").strip()
    tags = []
    for token in tagged.split():
        if "/" in token:
            tags.append(token.split("/")[-1])
    return tags


def extract_pos_feature_vector(pos: str, pos_dictionary) -> np.ndarray:

    vector = np.zeros(len(pos_dictionary))
    poses = re.split("[+-]", pos)
    idxs = []
    for pos in poses:
        if pos in pos_dictionary:
            idxs.append(pos_dictionary[pos])

    vector[idxs] = 1

    return vector
