import pathlib
import json
from enum import Enum

CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()

TOKENIZER_DATA_PATH = CONFIG_ROOT / 'tokenizer_data'

SPLIT_RATIO = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 4

class Dataset(str, Enum):
    DOCCI = 'docci'
    DOCCI_IIW = 'docci_iiw' # TODO
    COCO = 'coco'

DATASET = Dataset.COCO

class SpecialTokens(str, Enum):
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'

class EncoderArch(str, Enum):
    RESNET50 = 'resnet50'
    VIT_BASE_PATCH16_224 = 'vit-base-patch16-224'
    VIT_LARGE_PATCH16_224 = 'vit-large-patch16-224'
    CUSTOM_VIT_STYLE = 'custom-vit-style'
    # CUSTOM_SHOW_AND_TELL_STYLE = 'custom-show-attend-tell'

ENCODER_ARCH = EncoderArch.CUSTOM_VIT_STYLE # TODO

class MaxSeqLengthStrategy(str, Enum):
    MAX = 'max'
    PERCENTILE_90 = '90_percentile'
    PERCENTILE_92 = '92_percentile'
    PERCENTILE_95 = '95_percentile'
    PERCENTILE_99 = '99_percentile'
    CUSTOM = 'custom'

# Maximum text sequence length better be a power of 2 for efficiency

MAX_SEQUENCE_LENGTH_STRATEGY = MaxSeqLengthStrategy.CUSTOM

if MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.MAX:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / f'description_stats_{DATASET}.json'))['max']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_90:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / f'description_stats_{DATASET}.json'))['max_90']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_92:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / f'description_stats_{DATASET}.json'))['max_92']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_95:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / f'description_stats_{DATASET}.json'))['max_95']
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.PERCENTILE_99:
    MAX_TEXT_SEQUENCE_LENGTH = json.load(open(TOKENIZER_DATA_PATH / f'description_stats_{DATASET}.json'))['max_99']
else:
    MAX_TEXT_SEQUENCE_LENGTH = 60  # Custom value; adjust as needed

TOKENIZER_FILENAME_PREFIX = f'bpe_tokenizer_{DATASET}'

EVAL_DATASET_INFO_PATH= 'imageinwords/datasets' # TODO
EVAL_DATASET_SPLIT='IIW-400'

NUM_INPUT_CHANNELS = 3

# include bias in linear layers, except for last linear layer if weight tying is used
USE_BIAS = False

IMG_HEIGHT = 384
IMG_WIDTH = 384
PATCH_SIZE = 16
# NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)
IMG_EMBEDDING_DIM = 768 # doesn't need to be same as TEXT_EMBEDDING_DIM due to projection layer
USE_CONV_IMG_EMBEDDING = True

TEXT_VOCAB_SIZE = 1000 + SpecialTokens.__members__.__len__()  # adjust based on tokenizer vocab size
TEXT_EMBEDDING_DIM = 600 # our d_embed (different from d_model but in this case we set them the same)

EMBEDDING_DIM = TEXT_EMBEDDING_DIM  # shared embedding dimension for both image and text (d_model)

USE_PROJECTION_LAYER = (IMG_EMBEDDING_DIM != EMBEDDING_DIM)

USE_WEIGHT_TYING = True

ENCODER_NUM_BLOCKS = 12
ENCODER_NUM_HEADS = 12
ENCODER_DROPOUT_PROB = 0.1
ENCODER_HIDDEN_DIM = IMG_EMBEDDING_DIM * 4

DECODER_NUM_BLOCKS = 8
DECODER_NUM_HEADS = ENCODER_NUM_HEADS
DECODER_HIDDEN_DIM = EMBEDDING_DIM * 4
DECODER_DROPOUT_PROB = 0.1

NUM_EPOCHS = 7
SUBLAYER_DROPOUT = True
LR = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

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