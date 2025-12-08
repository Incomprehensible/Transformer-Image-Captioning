import pathlib
CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()

ENCODER_ARCHS = ['resnet50', 'vit-base-patch16-224', 'vit-large-patch16-224']

USE_EVAL_DATASET=True
EVAL_DATASET_INFO_PATH= 'imageinwords/datasets'
EVAL_DATASET_SPLIT='IIW-400'
