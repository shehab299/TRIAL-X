import torch
import torch.nn as nn


class InputWordDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):

        if not self.training or self.p == 0:
            return x

        x = x.clone()
        B, Ts, E = x.shape

        mask = torch.rand((B, Ts), device=x.device) < self.p
        mask = mask.unsqueeze(-1).expand(-1, -1, E)
        x[mask] = 0

        return x


class CharLevelEncoder(nn.Module):
    """
    Character-level encoder exactly as described in the D2 paper:
    - Takes all characters of a word as a sequence (Tw)
    - Char embedding: 32 dims
    - Word context: 512 dims (replicated Tw times)
    - Input to LSTM: 544 dims
    - 2-layer BiLSTM with hidden=512 → output=1024
    """

    def __init__(
        self,
        char_vocab_size=36,
        char_embed_dim=32,
        word_context_dim=512,
        hidden_size=512,
        num_layers=2,
    ):
        super().__init__()

        self.char_embed_dim = char_embed_dim  # 32
        self.word_context_dim = word_context_dim  # 512
        self.input_dim = char_embed_dim + word_context_dim  # 544
        self.hidden_size = hidden_size  # 512

        # (Tw,) → (Tw, 32)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)

        # BiLSTM: (Tw, 544) → (Tw, 1024)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, char_ids, word_context):
        """
        Inputs:
            char_ids:      (B, Tw) integer character IDs of words
            word_context:  (B, 2*Hw) vector f(w_i) from word-level encoder

        Output:
            char_contexts: (B, Tw, 1024)
        """
        _, Tw = char_ids.shape

        # 1) embed characters → (B, Tw, 32)
        char_emb = self.char_embedding(char_ids)

        # 2) repeat word context Tw times → (B, Tw, 2*Hw)
        wc_rep = word_context.unsqueeze(1).repeat(1, Tw, 1)

        # 3) concatenate along last dim → (B, Tw, 544)
        lstm_input = torch.cat([char_emb, wc_rep], dim=-1)

        # 4) BiLSTM → out: (B, Tw, 2*Hc)
        lstm_out, _ = self.lstm(lstm_input)

        return lstm_out


class EmbedDiacritizationModel(nn.Module):
    """
    DiacritizationModel: Word-level BiLSTM + Char-level encoder
    Inputs:
        x: (B, Ts, E) FloatTensor of word embeddings
        char_ids: (B, Ts, Tw) LongTensor of character IDs per word
    Outputs:
        char_contexts: (B, Ts, Tw, 2*Hc)
        word_contexts: (B, Ts, 2*Hw)
        logits: (B, Ts, Tw, num_classes)
    """

    def __init__(
        self,
        char_vocab_size=37,
        char_embed_dim=32,
        char_hidden_size=512,
        char_num_layers=2,
        word_embed_dim=300,  # 300 for FastText, 768 for AraBERT
        word_hidden_size=256,
        word_num_layers=2,
        num_classes=15,
    ):
        super().__init__()

        self.word_input_dropout = InputWordDropout(0.2)

        self.word_encoder = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=word_hidden_size,
            num_layers=word_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.word_level_dropout = nn.Dropout(p=0.2)

        self.char_level = CharLevelEncoder(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            word_context_dim=2 * word_hidden_size,
            hidden_size=char_hidden_size,
            num_layers=char_num_layers,
        )

        self.classifier = nn.Linear(2 * char_hidden_size, num_classes)

    def forward(self, word_embed, char_ids):
        """
        word_ids: word IDs per sentence in the batch (B, Ts, E)
        char_ids:  character IDs per word per sentence in the batch (B, Ts, Tw)
        """
        B, Ts, Tw = char_ids.shape

        word_embed = self.word_input_dropout(word_embed)
        word_context, _ = self.word_encoder(word_embed)  # (B, Ts, 2*Hw)
        word_context = self.word_level_dropout(word_context)  # Feature-Level Dropout
        word_context_flat = word_context.reshape(B * Ts, -1)  # (B*Ts, 2*Hw)

        flat_char_ids = char_ids.reshape(B * Ts, Tw)  # (B*Ts, Tw)
        char_reps = self.char_level(
            flat_char_ids, word_context_flat
        )  # (B*Ts, Tw, 2*Hc)
        char_contexts = char_reps.reshape(B, Ts, Tw, -1)  # (B, Ts, Tw, 2 * Hc)

        logits = self.classifier(char_contexts)  # (B, Ts, Tw, num_classes)

        return logits
