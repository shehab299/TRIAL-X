# ============================================
from tashkeel.preprocess import extract_letters_and_diacritics, segment_sentence
from tashkeel.features import extract_pos, extract_pos_feature_vector
from tashkeel.constants import posTagger

# ============================================
import torch
from torch.utils.data import Dataset
import numpy as np

# ============================================
import re

# ============================================


class DiacritizationDataset(Dataset):
    """
    Converts diacritized sentences into:
        - word IDs          (Ts,)
        - character IDs     (Ts, Tw)
        - diacritic labels  (Ts, Tw)
        - Pos Tags          (Ts, len(tags))

    Sentences are split into fixed-size segments of Ts words using a sliding
    window (stride). Words and characters are padded to fixed lengths.
    """

    def __init__(
        self,
        sentences,
        char2id,
        word2id,
        pos2id,
        pad_label=-100,
        Ts=10,
        stride=1,
        Tw=13,
    ):
        """
        Args:
            sentences: list of diacritized sentences.
            char2id: mapping char → ID (must include 'P').
            word2id: mapping undiacritized word → ID (<PAD>, <UNK> must exist).
            pad_label: label used for padding (default -100 for CE loss).
            Ts: number of words per segment.
            stride: step for segment sliding window.
            Tw: max characters per word.
        """
        self.sentences = sentences
        self.char2id = char2id
        self.word2id = word2id
        self.pos2id = pos2id
        self.pad_char = char2id["P"]
        self.pad_label = pad_label
        self.Ts = Ts
        self.Tw = Tw
        self.stride = stride

        # Precompute all segments as lists of words (strings) *but not tensors*
        self.segments = []
        for sentence in sentences:
            clean = re.sub(r"[^ء-ي\u064B-\u0652\s]", "", sentence)
            clean = re.sub(r"\s+", " ", clean).strip()
            words = clean.split()
            segments = segment_sentence(
                words, Ts, stride
            )  # returns list of list-of-strings
            self.segments.extend(segments)

    def __len__(self):
        """Return number of segments."""
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Returns:
            words  : (Ts,)     word IDs
            chars  : (Ts, Tw)  character IDs
            labels : (Ts, Tw)  diacritic labels
            pos:   : (Ts, len(tags)) hot-encoded vector of POS Tags
        """
        segment = self.segments[idx]  # list of Ts words (strings)
        chars = []
        diacritics = []
        words = []
        pos = []

        for word in segment:
            # 1) PAD word handling BEFORE cleaning
            if word == "<PAD>":
                word_id = self.word2id["<PAD>"]
                chars.append([self.pad_char] * self.Tw)
                diacritics.append([self.pad_label] * self.Tw)
                words.append(word_id)
                pos.append(torch.zeros(len(self.pos2id)))

                continue

            # 2) normal word
            undiacritized_word = re.sub(r"[^ء-ي]", "", word)
            word_id = self.word2id.get(undiacritized_word, self.word2id["<UNK>"])
            letters, labels = extract_letters_and_diacritics(word)
            ids = [self.char2id.get(ch, 0) for ch in letters]

            # pad to Tw (this assumes that the max word size is Tw i.e. no truncation might occur)
            if len(ids) < self.Tw:
                pad = self.Tw - len(ids)
                ids += [self.pad_char] * pad
                labels += [self.pad_label] * pad
            else:
                ids = ids[: self.Tw]
                labels = labels[: self.Tw]

            words.append(word_id)
            chars.append(ids)
            diacritics.append(labels)
            pos.append(
                extract_pos_feature_vector(extract_pos(word, posTagger)[0], self.pos2id)
            )

        return (
            torch.tensor(words, dtype=torch.long),
            torch.tensor(chars, dtype=torch.long),
            torch.tensor(diacritics, dtype=torch.long),
            torch.tensor(np.array(pos), dtype=torch.float),
        )


def collate_fn(batch):
    words, chars, diacs, pos = zip(*batch)
    return torch.stack(words), torch.stack(chars), torch.stack(diacs), torch.stack(pos)
