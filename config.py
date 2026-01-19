import pathlib
import json
from enum import Enum

# Folder structure
CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()
TOKENIZER_DATA_PATH = CONFIG_ROOT / 'tokenizer_data'

# Dataset config
SPLIT_RATIO = 0.2
RANDOM_SEED = 42

class Dataset(str, Enum):
    DOCCI = 'docci'
    DOCCI_IIW = 'docci_iiw' # TODO
    COCO = 'coco'
    FLICKR = 'flickr'

DATASET = Dataset.DOCCI

# Tokenizer config
class SpecialTokens(str, Enum):
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'

class TokenizerType(str, Enum):
    BPE = 'bpe'
    HF = 'hf'

TOKENIZER_TYPE = TokenizerType.BPE
# Upper bound on vocab size during tokenizer training
if DATASET == Dataset.COCO:
    UNIQUE_WORD_COUNT = 15000
elif DATASET == Dataset.FLICKR:
    UNIQUE_WORD_COUNT = 21000 # Estimation count of unique words in the dataset
elif DATASET == Dataset.DOCCI:
    UNIQUE_WORD_COUNT = 12000 # TODO

TOKENIZER_TRAIN_VOCAB_SIZE = UNIQUE_WORD_COUNT + SpecialTokens.__members__.__len__()
TOKENIZER_FILENAME_PREFIX = f'bpe_tokenizer_{DATASET}'

class MaxSeqLengthStrategy(str, Enum):
    MAX = 'max'
    PERCENTILE_90 = '90_percentile'
    PERCENTILE_92 = '92_percentile'
    PERCENTILE_95 = '95_percentile'
    PERCENTILE_99 = '99_percentile'
    CUSTOM = 'custom'

# Maximum text sequence length better be a power of 2 for efficiency

MAX_SEQUENCE_LENGTH_STRATEGY = MaxSeqLengthStrategy.PERCENTILE_99

if MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.MAX:
    path = TOKENIZER_DATA_PATH / f'max_desc_length_{DATASET}.json'
else:
    path = TOKENIZER_DATA_PATH / f'description_token_stats_{DATASET}.json'

if MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.MAX:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = json.load(open(path))['MAX_DESC_LENGTH']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_90:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = json.load(open(path))['max_90']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_92:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = json.load(open(path))['max_92']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_95:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = json.load(open(path))['max_95']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_99:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = json.load(open(path))['max_99']
else:
    MAX_TEXT_SEQUENCE_LENGTH = 60  # Custom value

# EVAL_DATASET_INFO_PATH= 'imageinwords/datasets' # TODO
# EVAL_DATASET_SPLIT='IIW-400'

# Model config

# Encoder architecture options
class EncoderArch(str, Enum):
    CUSTOM_CPTR_STYLE = 'custom-cptr-style'
    CNN_RESNET50 = 'resnet50'
    VIT_STYLE_BASE = 'google/vit-base-patch16-224-in21k'
    VIT_STYLE_LARGE = 'google/vit-large-patch16-224-in21k'
    CNN_CPTR_STYLE = 'cnn-cptr-style'
    # CUSTOM_SHOW_AND_TELL_STYLE = 'custom-show-attend-tell'

ENCODER_ARCH = EncoderArch.VIT_STYLE_BASE

class ViTEncodingStrategy(str, Enum):
    PATCHES = 'last_hidden_state_patches'
    CLS_TOKEN = 'cls_token'

# if ENCODER_ARCH == EncoderArch.VIT_STYLE_BASE or ENCODER_ARCH == EncoderArch.VIT_STYLE_LARGE:
VIT_ENCODING_STRATEGY = ViTEncodingStrategy.CLS_TOKEN

# Model hyperparameters
NUM_INPUT_CHANNELS = 3

# include bias in linear layers, except for last linear layer if weight tying is used
USE_BIAS = False

if ENCODER_ARCH == EncoderArch.VIT_STYLE_BASE or ENCODER_ARCH == EncoderArch.VIT_STYLE_LARGE:
  IMG_HEIGHT = 224
  IMG_WIDTH = 224
else: # custom
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)

if ENCODER_ARCH == EncoderArch.CUSTOM_CPTR_STYLE or ENCODER_ARCH == EncoderArch.CNN_CPTR_STYLE:
    IMG_EMBEDDING_DIM = PATCH_SIZE**2 * NUM_INPUT_CHANNELS # doesn't need to be same as TEXT_EMBEDDING_DIM due to projection layer
elif ENCODER_ARCH == EncoderArch.CNN_RESNET50:
    IMG_EMBEDDING_DIM = 2048  # ResNet-50 final feature map channels
elif ENCODER_ARCH == EncoderArch.VIT_STYLE_BASE:
    IMG_EMBEDDING_DIM = 768
elif ENCODER_ARCH == EncoderArch.VIT_STYLE_LARGE:
    IMG_EMBEDDING_DIM = 1024
else:
    IMG_EMBEDDING_DIM = 512  # default

USE_CONV_IMG_EMBEDDING = True

# text vocab size will be taken from tokenizer
TEXT_EMBEDDING_DIM = 512 # our d_embed (different from d_model but in this case we set them the same)

EMBEDDING_DIM = TEXT_EMBEDDING_DIM  # shared embedding dimension for both image and text (d_model)

USE_PROJECTION_LAYER = (IMG_EMBEDDING_DIM != EMBEDDING_DIM)

USE_WEIGHT_TYING = False

ENCODER_NUM_BLOCKS = 8
ENCODER_NUM_HEADS = 4
ENCODER_DROPOUT_PROB = 0.1
ENCODER_HIDDEN_DIM = IMG_EMBEDDING_DIM * 4

DECODER_NUM_BLOCKS = 8
DECODER_NUM_HEADS = 8
DECODER_HIDDEN_DIM = EMBEDDING_DIM * 4
DECODER_DROPOUT_PROB = 0.1
SUBLAYER_DROPOUT = False

# Training config
BATCH_SIZE_TRAIN = 24
BATCH_SIZE_VAL = 1
BATCH_SIZE_TEST = 16
NUM_FREEZE_EPOCHS = 7
NUM_EPOCHS = 2
LR = 3e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_DELTA = 0

# Params from CPTR paper:
# BATCH_SIZE         = 1              #(1)

# IMAGE_SIZE         = 384            #(2)
# IN_CHANNELS        = 3              #(3)

# SEQ_LENGTH         = 30             #(4)
# VOCAB_SIZE         = 10000          #(5)

# EMBED_DIM          = 768            #(6)
# PATCH_SIZE         = 16             #(7)
# NUM_PATCHES        = (IMAGE_SIZE//PATCH_SIZE) ** 2  #(8)
# NUM_ENCODER_BLOCKS = 12             #(9)
# NUM_DECODER_BLOCKS = 4              #(10)
# NUM_HEADS          = 12             #(11)
# HIDDEN_DIM         = EMBED_DIM * 4  #(12)
# DROP_PROB          = 0.1            #(13)