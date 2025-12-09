import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class AraBERTEmbedder:
    """
    AraBERTEmbedder: Contextual Arabic BERT-based Embedder

    Input: x: list of list of words, shape (B, Ts)
    Output: embeddings: FloatTensor of shape (B, Ts, 768)
    """

    def __init__(self, model_name="aubmindlab/bert-base-arabertv2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print(f"Loaded AraBERT model: {model_name} on {device}")

    def __call__(self, batch):
        return self._batch_to_tensor(batch)

    def _batch_to_tensor(self, batch):
        """
        batch: list of list of words, (B, Ts)
        """
        vectors = [self._sentence_to_tensor(words) for words in batch]
        return torch.stack(vectors)  # (B, Ts, 768)

    def _sentence_to_tensor(self, words):
        """
        Convert a list of words into contextual embeddings.
        """
        # tokenizer expects a list of words when is_split_into_words=True
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )

        # IMPORTANT: Keep the original encoding for word_ids before moving to device
        word_ids = encoding.word_ids(batch_index=0)

        # Now move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            last_hidden = outputs.last_hidden_state  # (1, N_subwords, 768)

        # For Ts words, collect their vectors
        Ts = len(words)
        final_vectors = []

        for word_index in range(Ts):
            # Find all subword positions belonging to this word
            positions = [i for i, wid in enumerate(word_ids) if wid == word_index]

            if len(positions) == 0:
                # Word was empty/padded: return zero vector
                vec = torch.zeros(768, device=self.device)
            else:
                # Average subword embeddings
                vec = last_hidden[0, positions].mean(dim=0)

            final_vectors.append(vec)

        # Stack directly as tensors (more efficient)
        result = torch.stack(final_vectors)  # (Ts, 768)

        # Safety: ensure no NaN/Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result
