from torch import nn
import torch
from torchcrf import CRF


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

        # Residual connection
        self.residual_proj = nn.Linear(self.input_dim, 2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

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

        # Residual connection
        res = self.residual_proj(lstm_input)
        lstm_out = self.layer_norm(lstm_out + res)

        return lstm_out


class CrossAttention(nn.Module):
    """
    Cross-attention layer that allows character representations to attend to word-level context.
    """

    def __init__(self, char_dim, word_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.char_dim = char_dim
        self.head_dim = char_dim // num_heads

        assert char_dim % num_heads == 0, "char_dim must be divisible by num_heads"

        # Query from character representations
        self.q_proj = nn.Linear(char_dim, char_dim)
        # Key and Value from word representations
        self.k_proj = nn.Linear(word_dim, char_dim)
        self.v_proj = nn.Linear(word_dim, char_dim)

        self.out_proj = nn.Linear(char_dim, char_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_reps, word_context):
        """
        char_reps: (B, Ts, Tw, char_dim) - character representations
        word_context: (B, Ts, word_dim) - word-level context

        Returns: (B, Ts, Tw, char_dim) - attended character representations
        """
        B, Ts, Tw, C = char_reps.shape

        # Reshape for multi-head attention
        # Query: from characters
        Q = self.q_proj(char_reps)  # (B, Ts, Tw, char_dim)
        Q = Q.reshape(B, Ts, Tw, self.num_heads, self.head_dim).permute(
            0, 1, 3, 2, 4
        )  # (B, Ts, H, Tw, head_dim)

        # Key, Value: from word context (expanded to attend from each character position)
        word_context_exp = word_context.unsqueeze(2)  # (B, Ts, 1, word_dim)
        K = self.k_proj(word_context_exp)  # (B, Ts, 1, char_dim)
        V = self.v_proj(word_context_exp)  # (B, Ts, 1, char_dim)

        K = K.reshape(B, Ts, 1, self.num_heads, self.head_dim).permute(
            0, 1, 3, 2, 4
        )  # (B, Ts, H, 1, head_dim)
        V = V.reshape(B, Ts, 1, self.num_heads, self.head_dim).permute(
            0, 1, 3, 2, 4
        )  # (B, Ts, H, 1, head_dim)

        # Attention scores: (B, Ts, H, Tw, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: (B, Ts, H, Tw, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: (B, Ts, Tw, char_dim)
        attn_output = attn_output.permute(0, 1, 3, 2, 4).reshape(B, Ts, Tw, C)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class DiacritizationModel(nn.Module):
    """
    DiacritizationModel: Word Level Embeddings + Word-level BiLSTM + Char-level encoder + Cross-Attention
    Inputs:
        x: (B, Ts) LongTensor of word IDs
        char_ids: (B, Ts, Tw) LongTensor of character IDs per word
        pos: (B, Ts, pos_tags_length) multi-hot encoded POS tags
    Outputs:
        logits: (B, Ts, Tw, num_classes)
    """

    def __init__(
        self,
        vocab_size,
        char_vocab_size=36,
        char_embed_dim=32,
        char_hidden_size=512,
        char_num_layers=2,
        word_embed_dim=300,
        word_hidden_size=256,
        word_num_layers=2,
        pos_tags_length=22,
        pad_idx=0,
        num_classes=15,
        use_cross_attention=True,
        num_attention_heads=8,
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=pad_idx)

        # Use linear projection for multi-hot POS tags
        self.pos_projection = nn.Linear(pos_tags_length, word_embed_dim)

        self.word_input_dropout = InputWordDropout(0.2)

        self.word_encoder = nn.LSTM(
            input_size=word_embed_dim * 2,  # word_embed + pos representation
            hidden_size=word_hidden_size,
            num_layers=word_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Residual connection for word encoder
        self.word_residual_proj = nn.Linear(word_embed_dim * 2, 2 * word_hidden_size)
        self.word_layer_norm = nn.LayerNorm(2 * word_hidden_size)

        self.word_level_dropout = nn.Dropout(p=0.2)

        self.char_level = CharLevelEncoder(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            word_context_dim=2 * word_hidden_size,
            hidden_size=char_hidden_size,
            num_layers=char_num_layers,
        )

        # Cross-attention layer
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(
                char_dim=2 * char_hidden_size,
                word_dim=2 * word_hidden_size,
                num_heads=num_attention_heads,
                dropout=0.1,
            )
            # Layer norm for residual connection
            self.layer_norm = nn.LayerNorm(2 * char_hidden_size)

        self.classifier = nn.Linear(2 * char_hidden_size, num_classes)

    def forward(self, word_ids, char_ids, pos):
        """
        word_ids: word IDs per sentence in the batch (B, Ts)
        char_ids: character IDs per word per sentence in the batch (B, Ts, Tw)
        pos: multi-hot encoded vector of POS tags (B, Ts, pos_tags_length)
        """
        B, Ts, Tw = char_ids.shape

        # ----- POS representation via linear projection -----
        pos_repr = self.pos_projection(pos.float())  # (B, Ts, word_embed_dim)

        # ------- Word embeddings ------
        word_embed = self.embedding(word_ids)  # (B, Ts, word_embed_dim)
        word_embed = torch.cat(
            [word_embed, pos_repr], dim=-1
        )  # (B, Ts, word_embed_dim * 2)

        word_embed = self.word_input_dropout(word_embed)  # (B, Ts, word_embed_dim * 2)

        word_context, _ = self.word_encoder(word_embed)  # (B, Ts, 2*Hw)

        # Residual connection
        res = self.word_residual_proj(word_embed)
        word_context = self.word_layer_norm(word_context + res)
        word_context = self.word_level_dropout(word_context)  # Feature-Level Dropout
        word_context_flat = word_context.reshape(B * Ts, -1)  # (B*Ts, 2*Hw)

        flat_char_ids = char_ids.reshape(B * Ts, Tw)  # (B*Ts, Tw)
        char_reps = self.char_level(
            flat_char_ids, word_context_flat
        )  # (B*Ts, Tw, 2*Hc)
        char_contexts = char_reps.reshape(B, Ts, Tw, -1)  # (B, Ts, Tw, 2 * Hc)

        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # Character representations attend to word-level context
            attn_output = self.cross_attention(
                char_contexts, word_context
            )  # (B, Ts, Tw, 2*Hc)
            # Residual connection + layer norm
            char_contexts = self.layer_norm(char_contexts + attn_output)

        logits = self.classifier(char_contexts)  # (B, Ts, Tw, num_classes)

        return logits


class DiacritizationModelWithCRF(nn.Module):
    """
    DiacritizationModel with CRF: Word Level Embeddings + Word-level BiLSTM + 
    Char-level encoder + Cross-Attention + CRF for sequence labeling
    
    Inputs:
        x: (B, Ts) LongTensor of word IDs
        char_ids: (B, Ts, Tw) LongTensor of character IDs per word
        pos: (B, Ts, pos_tags_length) multi-hot encoded POS tags
        targets: (B, Ts, Tw) LongTensor of target diacritic labels (only during training)
        mask: (B, Ts, Tw) BoolTensor indicating valid positions (optional)
        
    Outputs:
        During training: negative log-likelihood loss (scalar)
        During inference: predictions (B, Ts, Tw) with decoded sequences
    """

    def __init__(
        self,
        vocab_size,
        char_vocab_size=36,
        char_embed_dim=32,
        char_hidden_size=512,
        char_num_layers=2,
        word_embed_dim=300,
        word_hidden_size=256,
        word_num_layers=2,
        pos_tags_length=22,
        pad_idx=0,
        num_classes=15,
        use_cross_attention=True,
        num_attention_heads=8,
        use_crf=True,
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention
        self.use_crf = use_crf
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=pad_idx)

        # Use linear projection for multi-hot POS tags
        self.pos_projection = nn.Linear(pos_tags_length, word_embed_dim)

        self.word_input_dropout = InputWordDropout(0.2)

        self.word_encoder = nn.LSTM(
            input_size=word_embed_dim * 2,  # word_embed + pos representation
            hidden_size=word_hidden_size,
            num_layers=word_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Residual connection for word encoder
        self.word_residual_proj = nn.Linear(word_embed_dim * 2, 2 * word_hidden_size)
        self.word_layer_norm = nn.LayerNorm(2 * word_hidden_size)

        self.word_level_dropout = nn.Dropout(p=0.2)

        self.char_level = CharLevelEncoder(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            word_context_dim=2 * word_hidden_size,
            hidden_size=char_hidden_size,
            num_layers=char_num_layers,
        )

        # Cross-attention layer
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(
                char_dim=2 * char_hidden_size,
                word_dim=2 * word_hidden_size,
                num_heads=num_attention_heads,
                dropout=0.1,
            )
            # Layer norm for residual connection
            self.attn_layer_norm = nn.LayerNorm(2 * char_hidden_size)

        self.classifier = nn.Linear(2 * char_hidden_size, num_classes)
        
        # CRF layer for sequence labeling
        if self.use_crf:
            self.crf = CRF(num_classes, batch_first=True)

    def _compute_logits(self, word_ids, char_ids, pos):
        """
        Compute emission scores (logits) for CRF or direct classification.
        
        Returns:
            logits: (B, Ts, Tw, num_classes)
            word_context: (B, Ts, 2*Hw) for potential use
        """
        B, Ts, Tw = char_ids.shape

        # ----- POS representation via linear projection -----
        pos_repr = self.pos_projection(pos.float())  # (B, Ts, word_embed_dim)

        # ------- Word embeddings ------
        word_embed = self.embedding(word_ids)  # (B, Ts, word_embed_dim)
        word_embed = torch.cat(
            [word_embed, pos_repr], dim=-1
        )  # (B, Ts, word_embed_dim * 2)

        word_embed = self.word_input_dropout(word_embed)  # (B, Ts, word_embed_dim * 2)

        word_context, _ = self.word_encoder(word_embed)  # (B, Ts, 2*Hw)

        # Residual connection
        res = self.word_residual_proj(word_embed)
        word_context = self.word_layer_norm(word_context + res)
        word_context = self.word_level_dropout(word_context)  # Feature-Level Dropout
        word_context_flat = word_context.reshape(B * Ts, -1)  # (B*Ts, 2*Hw)

        flat_char_ids = char_ids.reshape(B * Ts, Tw)  # (B*Ts, Tw)
        char_reps = self.char_level(
            flat_char_ids, word_context_flat
        )  # (B*Ts, Tw, 2*Hc)
        char_contexts = char_reps.reshape(B, Ts, Tw, -1)  # (B, Ts, Tw, 2 * Hc)

        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # Character representations attend to word-level context
            attn_output = self.cross_attention(
                char_contexts, word_context
            )  # (B, Ts, Tw, 2*Hc)
            # Residual connection + layer norm
            char_contexts = self.attn_layer_norm(char_contexts + attn_output)

        logits = self.classifier(char_contexts)  # (B, Ts, Tw, num_classes)

        return logits, word_context

    def forward(self, word_ids, char_ids, pos, targets=None, mask=None):
        """
        Forward pass with CRF.
        
        Args:
            word_ids: (B, Ts) word IDs
            char_ids: (B, Ts, Tw) character IDs
            pos: (B, Ts, pos_tags_length) POS tags
            targets: (B, Ts, Tw) target labels (required during training)
            mask: (B, Ts, Tw) boolean mask for valid positions
            
        Returns:
            If training (targets provided): negative log-likelihood loss
            If inference: decoded predictions (B, Ts, Tw)
        """
        B, Ts, Tw = char_ids.shape
        
        # Compute emission scores
        logits, _ = self._compute_logits(word_ids, char_ids, pos)
        
        if not self.use_crf:
            # Without CRF, just return logits or compute standard cross-entropy
            if targets is not None:
                # Standard cross-entropy loss
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, self.num_classes),
                    targets.reshape(-1),
                    reduction='mean'
                )
                return loss
            else:
                return torch.argmax(logits, dim=-1)
        
        # Reshape for CRF: (B, Ts*Tw, num_classes)
        logits_flat = logits.reshape(B, Ts * Tw, self.num_classes)
        
        # Create or reshape mask for CRF
        if mask is None:
            # Create a mask that marks all positions as valid
            mask_flat = torch.ones(B, Ts * Tw, dtype=torch.bool, device=logits.device)
        else:
            mask_flat = mask.reshape(B, Ts * Tw)
        
        if self.training and targets is not None:
            # Training mode: compute negative log-likelihood
            targets_flat = targets.reshape(B, Ts * Tw)
            
            # CRF expects tags in [0, num_classes-1]. 
            # Replace -100 (padding) with 0 (or any valid index). 
            # The mask ensures these don't contribute to the loss.
            safe_targets = targets_flat.clone()
            safe_targets[~mask_flat] = 0
            
            # CRF forward returns negative log-likelihood
            # We want to minimize this, so return it as loss
            log_likelihood = self.crf(logits_flat, safe_targets, mask=mask_flat, reduction='mean')
            loss = -log_likelihood
            
            return loss
        else:
            # Inference mode: decode best sequence
            predictions_flat = self.crf.decode(logits_flat, mask=mask_flat)
            
            # Convert list of lists to tensor and reshape
            predictions = torch.tensor(predictions_flat, device=logits.device)
            predictions = predictions.reshape(B, Ts, Tw)
            
            return predictions
    
    def get_logits(self, word_ids, char_ids, pos):
        """
        Get raw logits without CRF decoding (useful for analysis).
        
        Returns:
            logits: (B, Ts, Tw, num_classes)
        """
        logits, _ = self._compute_logits(word_ids, char_ids, pos)
        return logits