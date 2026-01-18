from typing import List
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import config

from tokenizer import ByteLevelBPE

import math

# ==========================================================
# Encoder side 

class Patcher(torch.nn.Module):
    def __init__(self, patch_size=config.PATCH_SIZE, channels=config.NUM_INPUT_CHANNELS, emb_dim=config.IMG_EMBEDDING_DIM, bias=config.USE_BIAS):
        super(Patcher, self).__init__()
        self.P = patch_size
        self.C = channels
        self.D = emb_dim
        
        self.patcher = torch.nn.Unfold(kernel_size=self.P, stride=self.P)
        self.linear_embedding = torch.nn.Linear(in_features=self.P*self.P*self.C, out_features=self.D, bias=bias)
    
    def forward(self, images):
        if images.ndim < 4:
            images = images.unsqueeze(0)
        
        patches = self.patcher(images).permute(0, 2, 1)  # shape: (batch, num_patches, P*P*C)
        patches = self.linear_embedding(patches)  # shape: (batch, num_patches, D)
        return patches
    
    def get_linear_weights(self):
        return self.linear_embedding.weight

class ConvPatcher(torch.nn.Module):
    def __init__(self, patch_size=config.PATCH_SIZE, channels=config.NUM_INPUT_CHANNELS, emb_dim=config.IMG_EMBEDDING_DIM, bias=config.USE_BIAS, visualize_patches=False):
        super(ConvPatcher, self).__init__()
        self.P = patch_size
        self.C = channels
        self.D = emb_dim
        self.visualize = visualize_patches

        self.conv = torch.nn.Conv2d(in_channels=self.C, out_channels=self.D, kernel_size=self.P, stride=self.P, bias=bias, padding=0)

    def visualize_patches(self, convs, num=5):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 12))
        
        # Plot random image feature maps
        random_indexes = random.sample(range(0, self.D), k=num) # pick 5 numbers between 0 and the embedding size
        for i, idx in enumerate(random_indexes):
            image_conv_feature_map = convs[0, idx, :, :].squeeze().detach().cpu().numpy() # index on the output tensor of the convolutional layer
            axs[i].imshow(image_conv_feature_map, cmap='twilight')
            axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
    
    def forward(self, images):
        if images.ndim < 4:
            images = images.unsqueeze(0)
        
        # embedding dimension D becomes the number of output feature maps
        patches = self.conv(images)  # shape: (batch_size, embedding_dim, feature_map_height, feature_map_width)
        if self.visualize:
            self.visualize_patches(patches, num=5)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)  # shape: (batch_size, embedding_dim, num_patches)
        patches = patches.permute(0, 2, 1)
        return patches

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, num_patches, emb_dim=config.IMG_EMBEDDING_DIM):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = torch.nn.Parameter(requires_grad=True, data=torch.randn(size=(1, num_patches, emb_dim)))

    def forward(self):
        return self.pos_embedding

class EncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim=config.IMG_EMBEDDING_DIM, num_heads=config.ENCODER_NUM_HEADS, hidden_dim=config.ENCODER_HIDDEN_DIM, dropout_prob=config.ENCODER_DROPOUT_PROB, bias=config.USE_BIAS, sublayer_dropout=config.SUBLAYER_DROPOUT):
        super(EncoderBlock, self).__init__()
        self.MHSA = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_prob, bias=bias)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim, bias=bias),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(hidden_dim, embed_dim, bias=bias)
        )
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        
        if sublayer_dropout:
            self.sublayer_dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        attn_output, weights = self.MHSA(query=x, key=x, value=x)
        
        if hasattr(self, 'sublayer_dropout'):
            attn_output = self.sublayer_dropout(attn_output)
        
        x = self.layer_norm_1(x + attn_output)
        ff_output = self.FFN(x)
        x = self.layer_norm_2(x + ff_output)
        return x

# takes patches after linear projection and positional encoding
class Encoder(torch.nn.Module):
    def __init__(self, num_blocks=config.ENCODER_NUM_BLOCKS, embed_dim=config.IMG_EMBEDDING_DIM, num_heads=config.ENCODER_NUM_HEADS, hidden_dim=config.ENCODER_HIDDEN_DIM, dropout_prob=config.ENCODER_DROPOUT_PROB, bias=config.USE_BIAS, sublayer_dropout=config.SUBLAYER_DROPOUT):
        super(Encoder, self).__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.encoder_blocks.append(EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_prob=dropout_prob, bias=bias, sublayer_dropout=sublayer_dropout))

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x

# ==========================================================
# Decoder side

# <pad> does not contribute to loss, so we set padding_idx to its index in the vocabulary
class LearnableWordEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx):
        super(LearnableWordEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        
    def forward(self, input_ids):
        assert input_ids.dtype == torch.long, f"Input tensor must have dtype torch.long, got {input_ids.dtype}"
        embeddings = self.embedding(input_ids)
        return embeddings

class SinusoidPositionalEncoding(torch.nn.Module):
    def __init__(self,
        max_seq_len=config.MAX_TEXT_SEQUENCE_LENGTH,
        emb_dim=config.TEXT_EMBEDDING_DIM
    ):
        super().__init__()
        # create the positional encoding tensor of shape
        # maximum sequence length (L) by embedding dimension (D)
        pe = torch.zeros(max_seq_len, emb_dim, dtype=torch.float)

        # positions indexes: [0, 1, 2, ..., max_seq_len-1], with shape (L, 1)
        position = torch.arange(max_seq_len).unsqueeze(1)
        # frequency division terms with shape (D/2,) or (1, D/2)
        # use log for numerical stability: a**b = exp(b * log(a))
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000) / emb_dim)
        )

        # even positional encodings use sine, odd cosine
        # matrix-slice shape: (L, D/2), resulting matrix shape: (L, D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Buffers are for tensors that are not learnable parameters (no gradients) but are still part of the model's state
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor):
        # For regular inference, we don't need to pad the embeddings to max_seq_len anymore
        # Retrieve embeddings up to sequence length (S). output shape (1, S, C)
        return self.pe[:, :x.shape[1], :]

class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim=config.EMBEDDING_DIM, num_heads=config.DECODER_NUM_HEADS, hidden_dim=config.DECODER_HIDDEN_DIM, dropout_prob=config.DECODER_DROPOUT_PROB, bias=config.USE_BIAS, sublayer_dropout=config.SUBLAYER_DROPOUT, verbose=False):
        super(DecoderBlock, self).__init__()
        self.MMHSA = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_prob, bias=bias)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        # the bridge between the encoder and the decoder, K and V come from encoder, Q is derived from the previous decoder sublayer
        self.MHCA = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_prob / 2, bias=bias)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim, bias=bias),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(hidden_dim, embed_dim, bias=bias)
        )
        self.layer_norm_3 = torch.nn.LayerNorm(embed_dim)
        if sublayer_dropout:
            self.sublayer_dropout = torch.nn.Dropout(p=dropout_prob)
        self.verbose = verbose
        
        if verbose:
            print(f'DecoderBlock initialized with embed_dim={embed_dim}, num_heads={num_heads}, hidden_dim={hidden_dim}, dropout_prob={dropout_prob}, bias={bias}')

    # encoder output tensor will be passed as the key and value
    def forward(self, x, k, v, attn_mask, pad_mask):
        if x.ndim != 3:
            raise ValueError(f'Input tensor x must have 3 dimensions (batch_size, seq_length, embed_dim), but got {x.ndim} dimensions.')
        
        if self.verbose:
            print('Q shape:', x.shape)
            print('K shape:', k.shape)
        attn_output, mmhsa_w = self.MMHSA(query=x, key=x, value=x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        # print('Masked Multi-Head Self-Attention weights shape:', mmhsa_w.shape)
        
        if hasattr(self, 'sublayer_dropout'):
            # apply dropout before layer normalization for each sublayer
            attn_output = self.sublayer_dropout(attn_output)
            
        x = self.layer_norm_1(x + attn_output) # TODO: debug cross attention, and text vs img embeddings as inputs
        attn_output, mhca_w = self.MHCA(query=x, key=k, value=v)
        # print('Cross Attention weights shape:', mhca_w.shape)
        
        if hasattr(self, 'sublayer_dropout'):
            # apply dropout before layer normalization for each sublayer
            attn_output = self.sublayer_dropout(attn_output)
        
        x = self.layer_norm_2(x + attn_output)
        ff_output = self.FFN(x)
        x = self.layer_norm_3(x + ff_output)
        # print(f"Cross-attn weights mean: {mhca_w.mean()}, std: {mhca_w.std()}")
        if self.verbose:
            print(f'MMHSA weights shape: {mmhsa_w.shape}')
            print(f'MHCA weights shape: {mhca_w.shape}')
            print(f'FFN output shape: {ff_output.shape}')
            self.verbose = False  # only print once
        return x

class Decoder(torch.nn.Module):
    def __init__(self, num_blocks=config.DECODER_NUM_BLOCKS, embed_dim=config.EMBEDDING_DIM, num_heads=config.DECODER_NUM_HEADS, hidden_dim=config.DECODER_HIDDEN_DIM, dropout_prob=config.DECODER_DROPOUT_PROB, bias=config.USE_BIAS, sublayer_dropout=config.SUBLAYER_DROPOUT, verbose=False):
        super(Decoder, self).__init__()
        self.decoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_prob=dropout_prob, bias=bias, sublayer_dropout=sublayer_dropout, verbose=verbose))

    def forward(self, x, enc_output, attn_mask, pad_mask):
        for block in self.decoder_blocks:
            x = block.forward(x, k=enc_output, v=enc_output, attn_mask=attn_mask, pad_mask=pad_mask)
        return x

# ==========================================================
# Bridge / Projection layer

# project encoder output embeddings to the shared embedding dimension (which is based on text embedding dimension)
# is activated only if IMG_EMBEDDING_DIM != TEXT_EMBEDDING_DIM
class EmbeddingProjection(torch.nn.Module):
    def __init__(self, d_img_emb: int=config.IMG_EMBEDDING_DIM, d_model: int=config.TEXT_EMBEDDING_DIM, p_dropout=config.ENCODER_DROPOUT_PROB, bias=config.USE_BIAS):
        super(EmbeddingProjection, self).__init__()
        self.projection = torch.nn.Linear(in_features=d_img_emb, out_features=d_model, bias=bias)
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, x):
        # no need to scale by sqrt(d_model) here, as positional encodings were already added before the encoder input
        proj = self.projection(x)
        normed = self.layernorm(proj)
        dropped = self.dropout(normed)
        return dropped

# ==========================================================
# Full transformer

class CPTR(torch.nn.Module):
    def __init__(self, vocab_size, num_patches=(config.IMG_HEIGHT//config.PATCH_SIZE)*(config.IMG_WIDTH//config.PATCH_SIZE),
                 use_embedding_projection=config.USE_PROJECTION_LAYER,
                 img_emb_use_conv=config.USE_CONV_IMG_EMBEDDING,
                 img_emb_dim=config.IMG_EMBEDDING_DIM, 
                 patch_size=config.PATCH_SIZE, 
                 channels=config.NUM_INPUT_CHANNELS,
                 num_encoder_blocks=config.ENCODER_NUM_BLOCKS,
                 num_encoder_heads=config.ENCODER_NUM_HEADS,
                 encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
                 encoder_dropout_prob=config.ENCODER_DROPOUT_PROB,
                 text_emb_dim=config.TEXT_EMBEDDING_DIM,
                 d_model=config.EMBEDDING_DIM,
                 max_text_seq_len=config.MAX_TEXT_SEQUENCE_LENGTH,
                 pad_idx=0,
                 num_decoder_blocks=config.DECODER_NUM_BLOCKS,
                 num_decoder_heads=config.DECODER_NUM_HEADS,
                 decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
                 decoder_dropout_prob=config.DECODER_DROPOUT_PROB,
                 bias=config.USE_BIAS,
                 use_weight_tying=config.USE_WEIGHT_TYING,
                 sublayer_dropout=config.SUBLAYER_DROPOUT,
                 verbose=False):
        super(CPTR, self).__init__()
        
        # image side
        if img_emb_use_conv:
            self.patcher = ConvPatcher(emb_dim=img_emb_dim, patch_size=patch_size, channels=channels, bias=bias, visualize_patches=False)
        else:
            self.patcher = Patcher(patch_size=patch_size, channels=channels, emb_dim=img_emb_dim, bias=bias)
        self.img_pos_embedding = LearnablePositionalEmbedding(num_patches=num_patches, emb_dim=img_emb_dim)
        # encoder
        self.encoder = Encoder(num_blocks=num_encoder_blocks, 
                               embed_dim=img_emb_dim, 
                               num_heads=num_encoder_heads, 
                               hidden_dim=encoder_hidden_dim, 
                               dropout_prob=encoder_dropout_prob,
                               bias=bias,
                               sublayer_dropout=sublayer_dropout)
        
        if use_embedding_projection:
            # projection to shared embedding space
            self.emb_projector = EmbeddingProjection(d_img_emb=img_emb_dim, d_model=d_model, p_dropout=encoder_dropout_prob, bias=bias)
        
        # text side
        self.word_embedding = LearnableWordEmbedding(vocab_size=vocab_size, emb_dim=text_emb_dim, padding_idx=pad_idx)
        self.text_pos_embedding = SinusoidPositionalEncoding(max_seq_len=max_text_seq_len, emb_dim=d_model)
        assert d_model == text_emb_dim, f"In this implementation, d_model ({d_model}) must be equal to text_emb_dim ({text_emb_dim})"
        self.scaling = float(math.sqrt(d_model))
        self.text_layernorm = torch.nn.LayerNorm(d_model)
        self.text_dropout = torch.nn.Dropout(p=decoder_dropout_prob)
        # decoder
        self.decoder = Decoder(num_blocks=num_decoder_blocks, 
                               embed_dim=d_model, 
                               num_heads=num_decoder_heads, 
                               hidden_dim=decoder_hidden_dim, 
                               dropout_prob=decoder_dropout_prob,
                               bias=bias,
                               sublayer_dropout=sublayer_dropout,
                               verbose=verbose) # TODO: debug attention and masking
        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        
        # final linear + softmax layer
        # the output of the last decoder is used to predict the next word via a linear layer 
        # whose output dimension equals to the vocabulary size
        self.linear = torch.nn.Linear(in_features=d_model, out_features=vocab_size, bias=bias)
        if use_weight_tying:
            # W_out ​= W_embed^⊤
            self.linear.weight = self.word_embedding.embedding.weight # TODO: check dim
            
        self.softmax = torch.nn.LogSoftmax(dim=-1) # for inference mode # TODO

    def forward_images(self, images): # TODO: debug encoder having grad everywhere and changing
        patches = self.patcher(images)
        pos_emb = self.img_pos_embedding()
        emb = patches + pos_emb
        output = self.encoder(emb)
        if hasattr(self, 'emb_projector'): # TODO: enforce same dim
            output = self.emb_projector(output)
        # print('Image features shape:', output.shape)
        # print('Encoder output absolute mean', output.abs().mean()) # Should be around 0.5 - 1.5
        return output

    def forward_text(self, text_tokens, img_features, attn_mask=None, pad_mask=None):
        # ensure batch dimension
        if text_tokens.ndim < 2:
            text_tokens = text_tokens.unsqueeze(0)
        embeddings = self.word_embedding(text_tokens) * self.scaling # (B, L) -> (B, L, D)
        emb_sum = self.text_layernorm(embeddings + self.text_pos_embedding(embeddings))
        output = self.text_dropout(emb_sum)
        output = self.decoder(output, img_features, attn_mask, pad_mask) 
        return output
    
    def forward(self, images, text_tokens, attn_mask=None, pad_mask=None):
        img_features = self.forward_images(images) # K, V from encoder
        text_features = self.forward_text(text_tokens, img_features, attn_mask, pad_mask) # Q
        logits = self.linear(text_features)
        return logits

    @torch.inference_mode()
    def generate(self, 
                 image: torch.Tensor, 
                 bos_token: int,
                 eos_token: int,
                 max_len: int,
                 device: torch.device) -> List[int]:

        img_features = self.forward_images(image)

        tokens = torch.tensor(data=[[bos_token]], requires_grad=False).to(device)
        attn_mask = torch.triu(torch.ones((1, 1), device=device, requires_grad=False), diagonal=1).bool()

        while tokens.shape[1] < max_len and tokens[0, -1] != eos_token:
            text_features = self.forward_text(tokens, img_features, attn_mask, None) # Q
            logits = self.linear(text_features)
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            tokens = torch.cat(
                (tokens, torch.tensor([[next_token]], requires_grad=False).to(device)),
                dim = -1
            ).to(device)
            attn_mask = torch.triu(torch.ones((tokens.shape[1], tokens.shape[1]), device=device, requires_grad=False), diagonal=1).bool()
        return list(tokens[0])

    def forward_debug(self, images, text_tokens, attn_mask=None, pad_mask=None):
        img_features = self.forward_images(images) # K, V from encoder
        text_features = self.forward_text(text_tokens, img_features, attn_mask, pad_mask) # Q
        logits = self.linear(text_features)
        probs = self.softmax(logits) # TODO: check model outputs dim
        return probs, logits, img_features, text_features
        