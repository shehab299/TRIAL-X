import torch
import fasttext
import numpy as np


class FastTextEmbedder:
    """
    FastTextEmbedder: FastText Arabic Based Embedder
    Input:
        x: list of list of words, shape (B, Ts)
    Output:
        embeddings: FloatTensor of shape (B, Ts, E)
    """

    def __init__(self, fasttext_model=None, fasttext_path=None):
        if fasttext_model is not None:
            self.model = fasttext_model
        elif fasttext_path is not None:
            self.model = fasttext.load_model(fasttext_path)
        else:
            raise ValueError("Provide either fasttext_model or fasttext_path")

    def __call__(self, x):
        return self._batch_to_tensor(x)

    def _batch_to_tensor(self, batch):
        vectors = [self._sentence_to_tensor(sentence) for sentence in batch]
        return torch.stack(vectors)  # (B, Ts, E)

    def _sentence_to_tensor(self, words):
        vecs = []
        for w in words:
            if w == "" or w is None:
                # Return zero vector for padding
                vec = np.zeros(self.model.get_dimension())
            else:
                vec = self.model.get_word_vector(w)
            vecs.append(vec)

        result = torch.from_numpy(np.vstack(vecs)).float()  # (Ts, 300)

        # Safety check for NaN/Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"Warning: NaN/Inf in embeddings for words: {words}")
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result
