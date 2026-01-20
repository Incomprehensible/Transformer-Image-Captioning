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
TOKENIZER_FILENAME_PREFIX = f'bpe_tokenizer_{DATASET}'

class MaxSeqLengthStrategy(str, Enum):
    MAX = 'max'
    PERCENTILE_90 = '90_percentile'
    PERCENTILE_92 = '92_percentile'
    PERCENTILE_95 = '95_percentile'
    PERCENTILE_99 = '99_percentile'
    MEAN = 'mean'
    CUSTOM = 'custom'

# Maximum text sequence length better be a power of 2 for efficiency
MAX_SEQUENCE_LENGTH_STRATEGY = MaxSeqLengthStrategy.MEAN

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
elif MAX_SEQUENCE_LENGTH_STRATEGY == MaxSeqLengthStrategy.MEAN:
    if pathlib.Path(path).exists():
        MAX_TEXT_SEQUENCE_LENGTH = int(json.load(open(path))['mean'])
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
    CNN_CPTR_STYLE = 'cnn-cptr-style' # TODO
    # CUSTOM_SHOW_AND_TELL_STYLE = 'custom-show-attend-tell'

ENCODER_ARCH = EncoderArch.VIT_STYLE_BASE

class ViTEncodingStrategy(str, Enum):
    PATCHES = 'last_hidden_state_patches'
    CLS_TOKEN = 'cls_token'
    HYBRID = 'hybrid'

VIT_ENCODING_STRATEGY = ViTEncodingStrategy.HYBRID

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
TEXT_EMBEDDING_DIM = 768 # our d_embed (different from d_model but in this case we set them the same)

EMBEDDING_DIM = TEXT_EMBEDDING_DIM  # shared embedding dimension for both image and text (d_model)

USE_PROJECTION_LAYER = (IMG_EMBEDDING_DIM != EMBEDDING_DIM)

USE_WEIGHT_TYING = False

ENCODER_NUM_BLOCKS = 8
ENCODER_NUM_HEADS = 4
ENCODER_DROPOUT_PROB = 0.1
ENCODER_HIDDEN_DIM = IMG_EMBEDDING_DIM * 4

DECODER_NUM_BLOCKS = 10
DECODER_NUM_HEADS = 12
DECODER_HIDDEN_DIM = EMBEDDING_DIM * 4
DECODER_DROPOUT_PROB = 0.1
SUBLAYER_DROPOUT = True

# Training config
USE_ACCUMULATED_GRADIENTS = True
ACCUMULATION_STEPS = 4
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 1
BATCH_SIZE_TEST = 16
NUM_FREEZE_EPOCHS = 10
NUM_EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_DELTA = 0.005

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

# load currently defined config
def make_config():
    # export config as a json dictionary
    config_dict = {}
    for key, value in globals().items():
        if key.isupper():
            if isinstance(value, pathlib.PosixPath):
                config_dict[key] = str(value)
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
    return config_dict

def export_config(filepath: str):
    config_dict = make_config()
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)

def import_config(filepath: str):
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        globals()[key] = value
    return config_dict