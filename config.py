import pathlib
import json
from enum import Enum

CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()

TOKENIZER_DATA_PATH = CONFIG_ROOT / 'tokenizer_data'

class EncoderArch(str, Enum):
    RESNET50 = 'resnet50'
    VIT_BASE_PATCH16_224 = 'vit-base-patch16-224'
    VIT_LARGE_PATCH16_224 = 'vit-large-patch16-224'
    CUSTOM_VIT_STYLE = 'custom-vit-style'
    CUSTOM_SHOW_AND_TELL_STYLE = 'custom-show-attend-tell'

ENCODER_ARCH = EncoderArch.CUSTOM_VIT_STYLE

class MaxSeqLengthStrategy(str, Enum):
    MAX = 'max'
    PERCENTILE_90 = '90_percentile'
    PERCENTILE_92 = '92_percentile'
    PERCENTILE_95 = '95_percentile'
    PERCENTILE_99 = '99_percentile'
    CUSTOM = 'custom'

# Maximum text sequence length better be a power of 2 for efficiency

MAX_SEQUENCE_LENGTH_STRATEGY = MaxSeqLengthStrategy.PERCENTILE_90

if MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.MAX:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / 'description_stats.json'))['max']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_90:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / 'description_stats.json'))['max_90']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_92:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / 'description_stats.json'))['max_92']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_95:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / 'description_stats.json'))['max_95']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_99:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / 'description_stats.json'))['max_99']
else:
    MAX_TEXT_SEQUENCE_LENGTH = 1000  # Custom value; adjust as needed

USE_EVAL_DATASET=True
EVAL_DATASET_INFO_PATH= 'imageinwords/datasets'
EVAL_DATASET_SPLIT='IIW-400'

NUM_INPUT_CHANNELS = 3

IMG_HEIGHT = 224
IMG_WIDTH = 224
PATCH_SIZE = 16 #112 #16
# NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)

TEXT_VOCAB_SIZE = 1000
TEXT_EMBEDDING_DIM = 300 # our d_embed (different from d_model but in this case we set them the same)
IMG_EMBEDDING_DIM = 300 # doesn't need to be same as TEXT_EMBEDDING_DIM due to projection layer

USE_CONV_IMG_EMBEDDING = False

ENCODER_NUM_BLOCKS = 10
ENCODER_NUM_HEADS = 10
ENCODER_DROPOUT_PROB = 0.1
ENCODER_HIDDEN_DIM = IMG_EMBEDDING_DIM * 4

DECODER_DROPOUT_PROB = 0.1

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