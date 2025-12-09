import torch
import torch.nn as nn

class CWModel(nn.Module):
    """
    Core Word (CW) Model: Character-level BiLSTM.
    Predicts diacritics for all characters.
    """
    def __init__(
        self,
        char_vocab_size,
        char_embed_dim=32,
        hidden_size=256,
        num_layers=2,
        num_classes=15,
        dropout=0.2
    ):
        super().__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        """
        Args:
            char_ids: (B, Ts, Tw) or (B, TotalChars)
        Returns:
            logits: (B, Ts, Tw, num_classes)
        """
        # Flatten to (B * Ts, Tw) if input is 3D
        if char_ids.dim() == 3:
            B, Ts, Tw = char_ids.shape
            flat_char_ids = char_ids.reshape(B * Ts, Tw)
            is_3d = True
        else:
            flat_char_ids = char_ids
            is_3d = False
            
        x = self.char_embedding(flat_char_ids) # (N, Tw, E)
        x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x) # (N, Tw, 2*H)
        lstm_out = self.dropout(lstm_out)
        

        logits = self.classifier(lstm_out) # (N, Tw, C)
        
        if is_3d:
            logits = logits.reshape(B, Ts, Tw, -1)
            
        return logits
