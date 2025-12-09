import re
from typing import List
from tashkeel.constants import diacritics, diacritic2id


def extract_letters_and_diacritics(word):
    """
    Inputs:
    - word: string of diacritized arabic word.

    Outputs:
    - letters: list of characters (str) forming the word
    - labels: list of diacritics IDs (int)

    """
    letters = []
    labels = []
    i = 0
    while i < len(word):
        base = word[i]
        label = []
        j = i + 1
        while j < len(word) and word[j] in diacritics:
            label.append(word[j])
            j += 1

        # normalize: shadda always comes first if exists
        if "ّ" in label:
            label = ["ّ"] + [d for d in label if d != "ّ"]

        # combine to string
        label = "".join(label)

        # map to class
        label = diacritic2id.get(label, diacritic2id[""])

        letters.append(base)
        labels.append(label)
        i = j

    return letters, labels


def segment_sentence(sentence: List[str], Ts: int, stride: int) -> List[List[str]]:
    """
    Slides a window of length Ts and stride over a list of words.
    Output:
       - segments: list of lists of size Ts
    """
    segments = []
    n = len(sentence)

    for start in range(0, n, stride):

        end = min(start + Ts, n)
        seg = sentence[start:end]
        padding_needed = Ts - len(seg)
        seg += ["<PAD>"] * padding_needed

        segments.append(seg)

        if end >= n:
            break

    return segments


def build_vocab(sentences):
    """
    returns a dictionary mapping from word to its id
    """
    word2id = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        sentence = re.sub(r"[^ء-ي\s]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        tokens = sentence.split()
        for word in tokens:
            if word not in word2id:
                word2id[word] = len(word2id)
    return word2id


def segment_word_morphemes(segmenter, word):
    undiacritized = re.sub(r"[\u064B-\u0652]", "", word)
    segmented = segmenter.segment(undiacritized)
    morphemes = segmented.split("+")
    return morphemes


def split_sentence_morphemes(sentence, segmenter):
    tokens = []
    labels = []

    words = sentence.split(" ")

    for word in words:
        _, diacritics = extract_letters_and_diacritics(word)
        morphs = segment_word_morphemes(segmenter, word)

        i = 0
        for morph in morphs:
            tokens.append(morph)
            labels.append(diacritics[i : i + len(morph)])
            i += len(morph)

    return tokens, labels


def segment_sentence_morphemes(sentence, id2diacritic, segmenter):

    clean = re.sub(r"[^ء-ي\u064B-\u0652\s]", "", sentence)
    clean = re.sub(r"\s+", " ", clean)

    tokens, labels = split_sentence_morphemes(clean, segmenter)

    restored_tokens = []

    for token, label in zip(tokens, labels):
        new_token = ""
        for ch, d in zip(token, label):
            new_token += ch + id2diacritic[d]

        restored_tokens.append(new_token)

    return " ".join(restored_tokens).strip()
