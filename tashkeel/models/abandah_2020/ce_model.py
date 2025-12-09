import torch
import torch.nn as nn

class CEModel(nn.Module):
    """
    Case Ending (CE) Model: Word-level BiLSTM with rich features.
    Predicts diacritics for the last letter of each word (Case Ending).
    """
    def __init__(
        self,
        word_vocab_size,
        pos_feature_dim,
        gender_vocab_size=3, # N/A, M, F
        number_vocab_size=4, # N/A, S, D, P
        person_vocab_size=4, # N/A, 1, 2, 3
        word_embed_dim=300,
        hidden_size=256,
        num_layers=2,
        num_classes=15,
        dropout=0.2,
        pad_idx=0
    ):
        super().__init__()
        
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=pad_idx)
        
        # Explicit Feature Embeddings
        self.gender_embedding = nn.Embedding(gender_vocab_size, 4)
        self.number_embedding = nn.Embedding(number_vocab_size, 4)
        self.person_embedding = nn.Embedding(person_vocab_size, 4)
        
        # Project Legacy POS features
        self.pos_proj_dim = 30
        self.pos_projection = nn.Linear(pos_feature_dim, self.pos_proj_dim)
        
        # Total Input Dim
        input_dim = word_embed_dim + self.pos_proj_dim + 4 + 4 + 4
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, pos_features, gender_ids, number_ids, person_ids):
        """
        Args:
            word_ids: (B, Ts)
            pos_features: (B, Ts, F)
            gender_ids: (B, Ts)
            number_ids: (B, Ts)
            person_ids: (B, Ts)
        Returns:
            logits: (B, Ts, num_classes)
        """
        # Embed words
        word_emb = self.word_embedding(word_ids) # (B, Ts, E)
        
        # Embed Features
        gender_emb = self.gender_embedding(gender_ids) # (B, Ts, 4)
        number_emb = self.number_embedding(number_ids) # (B, Ts, 4)
        person_emb = self.person_embedding(person_ids) # (B, Ts, 4)
        
        # Project POS features
        pos_emb = self.pos_projection(pos_features) # (B, Ts, P)
        pos_emb = torch.relu(pos_emb)
        
        # Concatenate
        x = torch.cat([word_emb, pos_emb, gender_emb, number_emb, person_emb], dim=-1) # (B, Ts, Total)
        x = self.dropout(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x) # (B, Ts, 2*H)
        lstm_out = self.dropout(lstm_out)
        
        # Classify
        logits = self.classifier(lstm_out) # (B, Ts, C)
        
        return logits
