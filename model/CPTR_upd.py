from typing import List
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision.models import ResNet50_Weights
from transformers import ViTModel

import config

from tokenizer import ByteLevelBPE

import math

# ==========================================================
# Encoder side 

class Patcher(torch.nn.Module):
    def __init__(self, patch_size, 
                 channels, 
                 emb_dim, 
                 bias):
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
    def __init__(self, patch_size, 
                 channels, 
                 emb_dim, 
                 bias, 
                 visualize_patches=False):
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
        patches = patches.permute(0, 2, 1) # shape: (batch_size, num_patches, embedding_dim)
        return patches

class LearnablePositionalEmbedding(torch.nn.Module):
    def __init__(self, num_patches, emb_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = torch.nn.Parameter(requires_grad=True, data=torch.randn(size=(1, num_patches, emb_dim)))

    def forward(self):
        return self.pos_embedding

class CPTREncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, 
                 num_heads,
                 hidden_dim, 
                 dropout_prob, 
                 bias, 
                 sublayer_dropout):
        
        super(CPTREncoderBlock, self).__init__()
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
        residual = x

        x = self.layer_norm_1(x)

        attn_output, weights = self.MHSA(query=x, key=x, value=x)
        if hasattr(self, 'sublayer_dropout'):
            attn_output = self.sublayer_dropout(attn_output)
        
        x = residual + attn_output
        residual = x

        x = self.layer_norm_2(x)

        ff_output = self.FFN(x)
        if hasattr(self, 'sublayer_dropout'):
            ff_output = self.sublayer_dropout(ff_output)
        
        x = residual + ff_output

        return x

class CPTREncoder(torch.nn.Module):
    def __init__(self, img_emb_use_conv,
                 img_emb_dim, 
                 patch_size, 
                 channels,
                 num_patches,
                 num_blocks, 
                 num_heads, 
                 hidden_dim, 
                 dropout_prob, 
                 bias, 
                 sublayer_dropout):
        
        super(CPTREncoder, self).__init__()
        
        # image side
        if img_emb_use_conv:
            self.patcher = ConvPatcher(emb_dim=img_emb_dim, patch_size=patch_size, channels=channels, bias=bias, visualize_patches=False)
        else:
            self.patcher = Patcher(patch_size=patch_size, channels=channels, emb_dim=img_emb_dim, bias=bias)
        self.img_pos_embedding = LearnablePositionalEmbedding(num_patches=num_patches, emb_dim=img_emb_dim)
        
        self.encoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.encoder_blocks.append(CPTREncoderBlock(embed_dim=img_emb_dim, 
                                                        num_heads=num_heads, 
                                                        hidden_dim=hidden_dim, 
                                                        dropout_prob=dropout_prob, 
                                                        bias=bias, 
                                                        sublayer_dropout=sublayer_dropout))
        if sublayer_dropout:
            self.sublayer_dropout = torch.nn.Dropout(p=dropout_prob)
        
        self.images_norm = torch.nn.LayerNorm(img_emb_dim)

    def forward(self, x):
        x = self.patcher(x)

        x = x + self.img_pos_embedding()
        if hasattr(self, 'sublayer_dropout'):
            x = self.sublayer_dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x)

        x = self.images_norm(x)

        return x

class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the last two layers (Global Avg Pool and FC layer)
        # We want the 4th layer output (2048 channels)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        
        # ResNet50 output is [B, 2048, 7, 7] for a 224x224 image.
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        features = self.backbone(x) # [B, 2048, 7, 7]
        
        # Flatten the grid into a sequence of patches
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1) # [B, 49, 2048]
        
        return features # [B, 49, 2048]

class CNN_CPTREncoder(torch.nn.Module):
    def __init__(self,
                 img_emb_dim, 
                 num_patches,
                 num_blocks, 
                 num_heads, 
                 hidden_dim, 
                 dropout_prob, 
                 bias, 
                 sublayer_dropout):
        
        super(CNN_CPTREncoder, self).__init__()
        
        # image side
        self.patcher = CNNEncoder()
        self.img_pos_embedding = LearnablePositionalEmbedding(num_patches=num_patches, emb_dim=img_emb_dim)
        
        self.encoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.encoder_blocks.append(CPTREncoderBlock(embed_dim=img_emb_dim, 
                                                        num_heads=num_heads, 
                                                        hidden_dim=hidden_dim, 
                                                        dropout_prob=dropout_prob, 
                                                        bias=bias, 
                                                        sublayer_dropout=sublayer_dropout))
        if sublayer_dropout:
            self.sublayer_dropout = torch.nn.Dropout(p=dropout_prob)
        
        self.images_norm = torch.nn.LayerNorm(img_emb_dim)

    def forward(self, x):
        x = self.patcher(x)

        x = x + self.img_pos_embedding()
        if hasattr(self, 'sublayer_dropout'):
            x = self.sublayer_dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x)

        x = self.images_norm(x)

        return x

class ViTEncoder(torch.nn.Module):
    def __init__(self, model_patch, encoding_strategy, verbose=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_patch, output_attentions=True, output_hidden_states=False)
        
        # remove the classification head
        self.vit.heads = torch.nn.Identity()
        
        self.encoding_strategy = encoding_strategy
        self.verbose = verbose

    # ViT already contains final layernorm
    def forward(self, x, return_attn=False):
        # x: [B, 3, 224, 224]
        outputs = self.vit(x)
        # last_hidden_state: [Batch, 197, 768] (1 CLS + 196 patches)
        if self.encoding_strategy == config.ViTEncodingStrategy.CLS_TOKEN.value:
            features = outputs.last_hidden_state[:, 0, :] # CLS token
            features = features[:, None, :]
            if self.verbose:
                print(f'ViT Encoder using CLS token with shape: {features.shape}')  # [B, 1, 768]
        elif self.encoding_strategy == config.ViTEncodingStrategy.HYBRID.value:
            features = outputs.last_hidden_state  # [B, 197, 768]
        else: # PATCHES
            features = outputs.last_hidden_state[:, 1:, :] # # [B, 196, 768]
            if self.verbose:
                print(f'ViT Encoder using patch tokens with shape: {features.shape}')  # [B, 196, 768]
        
        if return_attn:
            return features, outputs.attentions
        return features

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
    def __init__(self, max_seq_len, emb_dim):
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
    def __init__(self, embed_dim, 
                 num_heads, 
                 hidden_dim, 
                 dropout_prob, 
                 bias, 
                 sublayer_dropout, 
                 verbose=False):
        
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

    # switched to pre-norm architecture
    # https://www.reddit.com/r/MachineLearning/comments/zzqzoy/d_does_it_make_sense_to_use_dropout_and_layer/
    # https://arxiv.org/pdf/2002.04745
    # encoder output tensor will be passed as the key and value
    def forward(self, x, kv, attn_mask, pad_mask):
        if x.ndim != 3:
            raise ValueError(f'Input tensor x must have 3 dimensions (batch_size, seq_length, embed_dim), but got {x.ndim} dimensions.')
        
        if self.verbose:
            print('Q shape:', x.shape)
            print('K/V shape:', kv.shape)
        
        residual = x
        x = self.layer_norm_1(x)
        attn_output, mmhsa_w = self.MMHSA(query=x, key=x, value=x, 
                                         attn_mask=attn_mask, 
                                         key_padding_mask=pad_mask)

        if hasattr(self, 'sublayer_dropout'):
            attn_output = self.sublayer_dropout(attn_output)
        x = residual + attn_output

        residual = x
        x = self.layer_norm_2(x)
        attn_output, mhca_w = self.MHCA(query=x, key=kv, value=kv)

        if hasattr(self, 'sublayer_dropout'):
            attn_output = self.sublayer_dropout(attn_output)
        x = residual + attn_output

        residual = x
        x = self.layer_norm_3(x)
        ff_output = self.FFN(x)

        if hasattr(self, 'sublayer_dropout'):
            ff_output = self.sublayer_dropout(ff_output)
        x = residual + ff_output
        
        if self.verbose:
            print(f"Cross-attn weights mean: {mhca_w.mean()}, std: {mhca_w.std()}")
            print(f'MMHSA weights shape: {mmhsa_w.shape}')
            print(f'MHCA weights shape: {mhca_w.shape}')
            print(f'FFN output shape: {ff_output.shape}')
            self.verbose = False  # only print once
        return x

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size,
                 text_emb_dim,
                 d_model,
                 max_text_seq_len,
                 pad_idx,
                 decoder_dropout_prob,
                 num_blocks, 
                 embed_dim, 
                 num_heads, 
                 hidden_dim,
                 dropout_prob, 
                 bias, 
                 sublayer_dropout, 
                 verbose=False):
        
        super(Decoder, self).__init__()
        
        self.word_embedding = LearnableWordEmbedding(vocab_size=vocab_size, emb_dim=text_emb_dim, padding_idx=pad_idx)
        self.text_pos_embedding = SinusoidPositionalEncoding(max_seq_len=max_text_seq_len, emb_dim=d_model)
        
        assert d_model == text_emb_dim, f"In this implementation, d_model ({d_model}) must be equal to text_emb_dim ({text_emb_dim})"
        
        self.scaling = float(math.sqrt(d_model))
        self.text_layernorm = torch.nn.LayerNorm(d_model)
        self.text_dropout = torch.nn.Dropout(p=decoder_dropout_prob)
        
        self.decoder_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(embed_dim=embed_dim, 
                                                    num_heads=num_heads, 
                                                    hidden_dim=hidden_dim, 
                                                    dropout_prob=dropout_prob, 
                                                    bias=bias, 
                                                    sublayer_dropout=sublayer_dropout, 
                                                    verbose=verbose))

    def forward(self, text_tokens, enc_output, attn_mask, pad_mask):
        # ensure batch dimension
        if text_tokens.ndim < 2:
            text_tokens = text_tokens.unsqueeze(0)
        
        x = self.word_embedding(text_tokens) * self.scaling # (B, L) -> (B, L, D)
        x = x + self.text_pos_embedding(text_tokens)
        x = self.text_dropout(x)
        
        for block in self.decoder_blocks:
            x = block.forward(x=x, kv=enc_output, attn_mask=attn_mask, pad_mask=pad_mask)

        x = self.text_layernorm(x)

        return x

# ==========================================================
# Bridge / Projection layer

# project encoder output embeddings to the shared embedding dimension (which is based on text embedding dimension)
# is activated only if IMG_EMBEDDING_DIM != TEXT_EMBEDDING_DIM
class EmbeddingProjection(torch.nn.Module):
    def __init__(self, d_img_emb: int, 
                 d_model: int,
                 p_dropout: float, 
                 bias: bool):
        
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
    def __init__(self, vocab_size,
                 encoder_arch,
                 encoding_strategy,
                 num_patches,
                 use_embedding_projection,
                 img_emb_use_conv,
                 img_emb_dim, 
                 patch_size, 
                 channels,
                 num_encoder_blocks,
                 num_encoder_heads,
                 encoder_hidden_dim,
                 encoder_dropout_prob,
                 text_emb_dim,
                 d_model,
                 max_text_seq_len,
                 pad_idx,
                 num_decoder_blocks,
                 num_decoder_heads,
                 decoder_hidden_dim,
                 decoder_dropout_prob,
                 bias,
                 use_weight_tying,
                 sublayer_dropout,
                 verbose=False):
        
        super(CPTR, self).__init__()
        self.ignore_index = pad_idx
        
        if encoder_arch == config.EncoderArch.CUSTOM_CPTR_STYLE:
            self.encoder = CPTREncoder(img_emb_use_conv=img_emb_use_conv,
                                   img_emb_dim=img_emb_dim, 
                                   patch_size=patch_size, 
                                   channels=channels,
                                   num_patches=num_patches,
                                   num_blocks=num_encoder_blocks, 
                                   num_heads=num_encoder_heads, 
                                   hidden_dim=encoder_hidden_dim, 
                                   dropout_prob=encoder_dropout_prob,
                                   bias=bias,
                                   sublayer_dropout=sublayer_dropout)
            print("Initialized CPTR Encoder")
        elif encoder_arch == config.EncoderArch.CNN_RESNET50:
            self.encoder = CNNEncoder()
            print("Initialized CNN ResNet-50 Encoder")
        elif encoder_arch == config.EncoderArch.VIT_STYLE_BASE or \
            encoder_arch == config.EncoderArch.VIT_STYLE_LARGE:
            model_patch = encoder_arch
            self.encoder = ViTEncoder(model_patch=model_patch, encoding_strategy=encoding_strategy)
            print(f"Initialized ViT Encoder: {model_patch}")
        elif encoder_arch == config.EncoderArch.CNN_CPTR_STYLE:
            self.encoder = CNN_CPTREncoder(img_emb_dim=img_emb_dim, 
                                   num_patches=num_patches,
                                   num_blocks=num_encoder_blocks, 
                                   num_heads=num_encoder_heads, 
                                   hidden_dim=encoder_hidden_dim, 
                                   dropout_prob=encoder_dropout_prob,
                                   bias=bias,
                                   sublayer_dropout=sublayer_dropout)
            print("Initialized CNN + CPTR Encoder")
        else:
            raise ValueError(f"Unsupported encoder architecture: {encoder_arch}")
        
        if use_embedding_projection:
            # projection to shared embedding space
            self.emb_projector = EmbeddingProjection(d_img_emb=img_emb_dim, d_model=d_model, p_dropout=encoder_dropout_prob, bias=bias)
        
        self.decoder = Decoder(vocab_size=vocab_size,
                               text_emb_dim=text_emb_dim,
                               d_model=d_model,
                               max_text_seq_len=max_text_seq_len,
                               pad_idx=pad_idx,
                               decoder_dropout_prob=decoder_dropout_prob,
                               num_blocks=num_decoder_blocks, 
                               embed_dim=d_model, 
                               num_heads=num_decoder_heads, 
                               hidden_dim=decoder_hidden_dim, 
                               dropout_prob=decoder_dropout_prob,
                               bias=bias,
                               sublayer_dropout=sublayer_dropout,
                               verbose=verbose)
        
        # final linear + softmax layer
        # the output of the last decoder is used to predict the next word via a linear layer 
        # whose output dimension equals to the vocabulary size
        self.linear = torch.nn.Linear(in_features=d_model, out_features=vocab_size, bias=bias)
        if use_weight_tying:
            # W_out ​= W_embed^⊤
            self.linear.weight = self.decoder.word_embedding.embedding.weight
        
        # Apply Xavier initialization to linear layers
        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        
        self.softmax = torch.nn.LogSoftmax(dim=-1) # for inference mode # TODO
    
    def forward(self, images, text_tokens, attn_mask=None, pad_mask=None, targets=None):
        img_features = self.encoder(images) # K, V from encoder
        
        if hasattr(self, 'emb_projector'):
            img_features = self.emb_projector(img_features)
            
        text_features = self.decoder(text_tokens=text_tokens, 
                                     enc_output=img_features, 
                                     attn_mask=attn_mask, 
                                     pad_mask=pad_mask)
        logits = self.linear(text_features)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=self.ignore_index)
        
        return logits, loss

    @torch.inference_mode()
    def generate(self, 
                 image: torch.Tensor, 
                 bos_token: int,
                 eos_token: int,
                 max_len: int,
                 device: torch.device) -> List[int]:

        img_features = self.encoder(image)
        
        if hasattr(self, 'emb_projector'):
            img_features = self.emb_projector(img_features)

        tokens = torch.tensor(data=[[bos_token]], requires_grad=False).to(device)
        attn_mask = torch.triu(torch.ones((1, 1), device=device, requires_grad=False), diagonal=1).bool()

        while tokens.shape[1] < max_len and tokens[0, -1] != eos_token:
            text_features = self.decoder(text_tokens=tokens, enc_output=img_features, attn_mask=attn_mask, pad_mask=None) # Q
            logits = self.linear(text_features)
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            tokens = torch.cat(
                (tokens, torch.tensor([[next_token]], requires_grad=False).to(device)),
                dim = -1
            ).to(device)
            attn_mask = torch.triu(torch.ones((tokens.shape[1], tokens.shape[1]), device=device, requires_grad=False), diagonal=1).bool()
        return list(tokens[0])

        