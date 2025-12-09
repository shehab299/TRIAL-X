import torch
from torch.utils.data import Dataset
import numpy as np
import re
from tashkeel.preprocess import extract_letters_and_diacritics, segment_sentence
from tashkeel.features import extract_pos, extract_pos_feature_vector
from tashkeel.constants import posTagger

class AbandahDataset(Dataset):
    """
    Dataset for Abandah et al. (2020) model.
    Returns:
        - word_ids: (Ts,)
        - char_ids: (Ts, Tw)
        - diacritics: (Ts, Tw) - Full diacritics
        - pos_features: (Ts, num_pos_features) - Multi-hot POS/Gender/Number (Legacy/Extra)
        - gender_ids: (Ts,)
        - number_ids: (Ts,)
        - person_ids: (Ts,)
        - word_lengths: (Ts,) - Index of the last valid character (for CE)
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
        self.sentences = sentences
        self.char2id = char2id
        self.word2id = word2id
        self.pos2id = pos2id
        self.pad_char = char2id["P"]
        self.pad_label = pad_label
        self.Ts = Ts
        self.Tw = Tw
        self.stride = stride

        # Mappings for explicit features
        self.gender_map = {"M": 1, "F": 2} # 0 is N/A
        self.number_map = {"S": 1, "D": 2, "P": 3} # 0 is N/A
        self.person_map = {"1": 1, "2": 2, "3": 3} # 0 is N/A

        # Precompute segments
        self.segments = []
        for sentence in sentences:
            clean = re.sub(r"[^ء-ي\u064B-\u0652\s]", "", sentence)
            clean = re.sub(r"\s+", " ", clean).strip()
            words = clean.split()
            if not words:
                continue
            segments = segment_sentence(words, Ts, stride)
            self.segments.extend(segments)

    def __len__(self):
        return len(self.segments)
    
    def extract_morph_features(self, tag):
        """
        Extracts Gender, Number, Person from Farasa tag string.
        Example tag: "DET+NOUN-MS" -> Gender=M, Number=S
        """
        g, n, p = 0, 0, 0
        
        if not tag:
            return g, n, p
            
        # Gender
        if "-M" in tag or "+M" in tag or "MS" in tag or "MP" in tag or "MD" in tag:
            g = self.gender_map["M"]
        elif "-F" in tag or "+F" in tag or "FS" in tag or "FP" in tag or "FD" in tag:
            g = self.gender_map["F"]
            
        # Number
        if "S" in tag and ("MS" in tag or "FS" in tag or "-S" in tag):
            n = self.number_map["S"]
        elif "D" in tag and ("MD" in tag or "FD" in tag or "-D" in tag):
            n = self.number_map["D"]
        elif "P" in tag and ("MP" in tag or "FP" in tag or "-P" in tag):
            n = self.number_map["P"]
            
        # Person (usually in verbs like V-1P)
        if "1" in tag:
            p = self.person_map["1"]
        elif "2" in tag:
            p = self.person_map["2"]
        elif "3" in tag:
            p = self.person_map["3"]
            
        return g, n, p

    def __getitem__(self, idx):
        segment = self.segments[idx]
        
        word_ids_list = []
        char_ids_list = []
        diacritics_list = []
        pos_features_list = []
        gender_list = []
        number_list = []
        person_list = []
        word_lengths_list = []

        for word in segment:
            # Handle PAD word
            if word == "<PAD>":
                word_ids_list.append(self.word2id["<PAD>"])
                char_ids_list.append([self.pad_char] * self.Tw)
                diacritics_list.append([self.pad_label] * self.Tw)
                pos_features_list.append(torch.zeros(len(self.pos2id)))
                gender_list.append(0)
                number_list.append(0)
                person_list.append(0)
                word_lengths_list.append(0)
                continue

            # Normal word
            undiacritized_word = re.sub(r"[^ء-ي]", "", word)
            word_id = self.word2id.get(undiacritized_word, self.word2id["<UNK>"])
            word_ids_list.append(word_id)

            letters, labels = extract_letters_and_diacritics(word)
            c_ids = [self.char2id.get(ch, 0) for ch in letters]
            
            valid_len = len(c_ids)
            if valid_len > self.Tw:
                valid_len = self.Tw
                c_ids = c_ids[:self.Tw]
                labels = labels[:self.Tw]
            
            pad_len = self.Tw - len(c_ids)
            c_ids += [self.pad_char] * pad_len
            labels += [self.pad_label] * pad_len
            
            char_ids_list.append(c_ids)
            diacritics_list.append(labels)
            word_lengths_list.append(valid_len)

            # Features
            try:
                tags = extract_pos(word, posTagger)
                tag = tags[0] if tags else ""
            except:
                tag = ""
            
            pos_vec = extract_pos_feature_vector(tag, self.pos2id)
            pos_features_list.append(pos_vec)
            
            g, n, p = self.extract_morph_features(tag)
            gender_list.append(g)
            number_list.append(n)
            person_list.append(p)

        return (
            torch.tensor(word_ids_list, dtype=torch.long),
            torch.tensor(char_ids_list, dtype=torch.long),
            torch.tensor(diacritics_list, dtype=torch.long),
            torch.tensor(np.array(pos_features_list), dtype=torch.float),
            torch.tensor(gender_list, dtype=torch.long),
            torch.tensor(number_list, dtype=torch.long),
            torch.tensor(person_list, dtype=torch.long),
            torch.tensor(word_lengths_list, dtype=torch.long),
        )

def collate_fn(batch):
    words, chars, diacs, pos, gender, number, person, lens = zip(*batch)
    return (
        torch.stack(words),
        torch.stack(chars),
        torch.stack(diacs),
        torch.stack(pos),
        torch.stack(gender),
        torch.stack(number),
        torch.stack(person),
        torch.stack(lens)
    )
