# =============================================
from torch.utils.data import Dataset
import torch
from farasa import FarasaSegmenter

# =============================================
from tashkeel.preprocess import extract_letters_and_diacritics, segment_sentence
from tashkeel.constants import id2diacritic, diacritic2id

# =============================================
import re

# =============================================


class DiacritizationDataset(Dataset):
    """
    Converts diacritized sentences into:
        - Word Embeddings   (Ts, E)
        - character IDs     (Ts, Tw)
        - diacritic labels  (Ts, Tw)

    Sentences are split into fixed-size segments of Ts words using a sliding
    window (stride). Words and characters are padded to fixed lengths.
    """

    def __init__(
        self,
        sentences,
        char2id,
        pad_label=-100,
        Ts=10,
        stride=1,
        Tw=13,
    ):
        """
        Args:
            sentences: list of diacritized sentences.
            char2id: mapping char → ID (must include 'P').
            pad_label: label used for padding (default -100 for CE loss).
            Ts: number of words per segment.
            stride: step for segment sliding window.
            Tw: max characters per word.
        """
        self.sentences = sentences
        self.char2id = char2id
        self.pad_char = char2id["P"]
        self.pad_label = pad_label
        self.Ts = Ts
        self.Tw = Tw
        self.stride = stride

        # Precompute all segments as lists of words (strings) *but not tensors*
        self.segments = []
        for sentence in sentences:
            clean = re.sub(r"[^ء-ي\u064B-\u0652\s]", "", sentence)
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
            words  : list[str]   list of Ts words
            chars  : (Ts, Tw)  character IDs
            labels : (Ts, Tw)  diacritic labels
        """
        segment = self.segments[idx]  # list of Ts words (strings)
        chars = []
        diacritics = []
        words = []
        for word in segment:
            # 1) PAD word handling BEFORE cleaning
            if word == "<PAD>":
                chars.append([self.pad_char] * self.Tw)
                diacritics.append([diacritic2id[""]] * self.Tw)
                words.append("")  # Empty string for padding
                continue

            # 2) normal word
            undiacritized_word = re.sub(r"[^ء-ي]", "", word)
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

            words.append(undiacritized_word)
            chars.append(ids)
            diacritics.append(labels)

        return (
            words,  # Return list of words as strings
            torch.tensor(chars, dtype=torch.long),
            torch.tensor(diacritics, dtype=torch.long),
        )


def collate_fn(batch, embedder):
    """
    Custom collate function for DiacritizationDatasetEmbed.
    Takes batch of (words_list, chars, diacs) and embeds the words using FastText.

    Args:
        batch: List of tuples (words, chars, diacs) from dataset
        embedder: FastTextEmbedder instance

    Returns:
        word_embeddings: (B, Ts, E) FloatTensor
        chars: (B, Ts, Tw) LongTensor
        diacs: (B, Ts, Tw) LongTensor
    """
    words_batch, chars_batch, diacs_batch = zip(*batch)

    # Embed all sentences in the batch
    word_embeddings = embedder(list(words_batch))  # (B, Ts, E)

    # Stack the character and diacritic tensors
    chars = torch.stack(chars_batch)  # (B, Ts, Tw)
    diacs = torch.stack(diacs_batch)  # (B, Ts, Tw)

    return word_embeddings, chars, diacs
