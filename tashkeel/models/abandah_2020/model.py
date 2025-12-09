import torch
import torch.nn as nn
from .cw_model import CWModel
from .ce_model import CEModel

class AbandahModel(nn.Module):
    """
    Abandah et al. (2020) Model Wrapper.
    Combines CW (Core Word) and CE (Case Ending) models.
    """
    def __init__(
        self,
        char_vocab_size,
        word_vocab_size,
        pos_feature_dim,
        char_embed_dim=32,
        char_hidden_size=256,
        char_num_layers=2,
        word_embed_dim=300,
        word_hidden_size=256,
        word_num_layers=2,
        num_classes=15,
        pad_idx=0
    ):
        super().__init__()
        
        self.cw_model = CWModel(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            hidden_size=char_hidden_size,
            num_layers=char_num_layers,
            num_classes=num_classes
        )
        
        self.ce_model = CEModel(
            word_vocab_size=word_vocab_size,
            pos_feature_dim=pos_feature_dim,
            word_embed_dim=word_embed_dim,
            hidden_size=word_hidden_size,
            num_layers=word_num_layers,
            num_classes=num_classes,
            pad_idx=pad_idx
        )
        
        self.num_classes = num_classes

    def forward(self, word_ids, char_ids, pos_features, gender_ids, number_ids, person_ids, word_lengths):
        """
        Args:
            word_ids: (B, Ts)
            char_ids: (B, Ts, Tw)
            pos_features: (B, Ts, F)
            gender_ids: (B, Ts)
            number_ids: (B, Ts)
            person_ids: (B, Ts)
            word_lengths: (B, Ts)
        Returns:
            logits: (B, Ts, Tw, num_classes)
        """
        # 1. Run CW Model
        cw_logits = self.cw_model(char_ids) # (B, Ts, Tw, C)
        
        # 2. Run CE Model
        ce_logits = self.ce_model(word_ids, pos_features, gender_ids, number_ids, person_ids) # (B, Ts, C)
        
        # 3. Merge Outputs
        B, Ts, Tw, C = cw_logits.shape
        
        last_char_indices = word_lengths - 1
        last_char_indices = last_char_indices.clamp(min=0)
        
        flat_logits = cw_logits.clone().reshape(B * Ts, Tw, C)
        flat_ce_logits = ce_logits.reshape(B * Ts, C)
        flat_indices = last_char_indices.reshape(B * Ts)
        
        batch_indices = torch.arange(B * Ts, device=cw_logits.device)
        
        valid_mask = word_lengths.reshape(B * Ts) > 0
        
        valid_batch_indices = batch_indices[valid_mask]
        valid_char_indices = flat_indices[valid_mask]
        
        flat_logits[valid_batch_indices, valid_char_indices] = flat_ce_logits[valid_mask]
        
        final_logits = flat_logits.reshape(B, Ts, Tw, C)
        
        return final_logits
