import pathlib
CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()

ENCODER_ARCHS = ['resnet50', 'vit-base-patch16-224', 'vit-large-patch16-224']

USE_EVAL_DATASET=True
EVAL_DATASET_INFO_PATH= 'imageinwords/datasets'
EVAL_DATASET_SPLIT='IIW-400'

NUM_INPUT_CHANNELS = 3

IMG_HEIGHT = 224
IMG_WIDTH = 224
PATCH_SIZE = 16 #112 #16
# NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)

TEXT_VOCAB_SIZE = 1000
TEXT_EMBEDDING_DIM = 300
TEXT_MAX_CAPTION_LEN = 500 # needs to be adjusted based on dataset
IMG_EMBEDDING_DIM = 300

USE_CONV_IMG_EMBEDDING = False

ENCODER_NUM_BLOCKS = 10
ENCODER_NUM_HEADS = 10
ENCODER_DROPOUT_PROB = 0.1
ENCODER_HIDDEN_DIM = IMG_EMBEDDING_DIM * 4

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