import pathlib
import torch
from tqdm.auto import tqdm
from statistics import mean
import json
import config as cfg

from dataset.loader import DatasetLoader
from tokenizer.tokenizer import ByteLevelBPE, TokenizerHF
from model.CPTR_upd import CPTR
from evaluation.nlg_metrics import Metrics
import random

import argparse


def tokenize(sentence: str):
    return sentence.lower().strip().split()


def evaluate(model_folder: str, model_name: str = None, num_batches: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_folder = pathlib.Path(model_folder)
    config = cfg.import_config(model_folder / "config.json")
    
    if model_name is None:
        model_path = model_folder / "cptr_model.pth"
    else:
        if not '.pth' in model_name:
            model_name += ".pth"
        model_path = model_folder / model_name

    data_loader = DatasetLoader(
        dataset_type=config["DATASET"],
        img_height=config["IMG_HEIGHT"],
        img_width=config["IMG_WIDTH"],
        batch_size_train=cfg.BATCH_SIZE_TRAIN,
        batch_size_test=cfg.BATCH_SIZE_TEST,
        split_ratio=config["SPLIT_RATIO"],
        shuffle_test=False,
        seed=config["RANDOM_SEED"]
    )
    data_loader.load_data()
    test_dataloader = data_loader.get_test_dataloader()

    special_tokens = [cfg.SpecialTokens.PAD, cfg.SpecialTokens.BOS, cfg.SpecialTokens.EOS]

    if config["TOKENIZER_TYPE"] == cfg.TokenizerType.HF:
        tokenizer = TokenizerHF()
    else:
        tokenizer = ByteLevelBPE(special_tokens=special_tokens)
        tokenizer.load(
            folder=cfg.TOKENIZER_DATA_PATH,
            filename_prefix=config["TOKENIZER_FILENAME_PREFIX"]
        )

    pad_idx = tokenizer.get_padding_token_id()
    vocab_size = tokenizer.get_vocab_size()

    model = CPTR(
        num_patches=config["NUM_PATCHES"],
        encoder_arch=config["ENCODER_ARCH"],
        encoding_strategy=config["VIT_ENCODING_STRATEGY"],
        use_embedding_projection=config["USE_PROJECTION_LAYER"],
        img_emb_use_conv=config["USE_CONV_IMG_EMBEDDING"],
        img_emb_dim=config["IMG_EMBEDDING_DIM"],
        patch_size=config["PATCH_SIZE"],
        text_emb_dim=config["TEXT_EMBEDDING_DIM"],
        d_model=config["EMBEDDING_DIM"],
        max_text_seq_len=config["MAX_TEXT_SEQUENCE_LENGTH"],
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        channels=config["NUM_INPUT_CHANNELS"],
        num_encoder_blocks=config["ENCODER_NUM_BLOCKS"],
        num_encoder_heads=config["ENCODER_NUM_HEADS"],
        encoder_hidden_dim=config["ENCODER_HIDDEN_DIM"],
        encoder_dropout_prob=config["ENCODER_DROPOUT_PROB"],
        num_decoder_blocks=config["DECODER_NUM_BLOCKS"],
        num_decoder_heads=config["DECODER_NUM_HEADS"],
        decoder_hidden_dim=config["DECODER_HIDDEN_DIM"],
        decoder_dropout_prob=config["DECODER_DROPOUT_PROB"],
        bias=config["USE_BIAS"],
        use_weight_tying=config["USE_WEIGHT_TYING"],
        sublayer_dropout=config["SUBLAYER_DROPOUT"],
        verbose=False
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics = Metrics()
    all_refs = []
    all_hypos = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch["pixel_values"].to(device)
            gt_captions = batch["description"]

            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)

                tokens = model.generate(
                    image,
                    bos_token=tokenizer.get_vocab()[cfg.SpecialTokens.BOS.value],
                    eos_token=tokenizer.get_vocab()[cfg.SpecialTokens.EOS.value],
                    max_len=config["MAX_TEXT_SEQUENCE_LENGTH"],
                    device=device,
                    used_tokens_penalty=True
                )

                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens)

                pred_caption = tokenizer.decode(tokens)
                pred_caption = tokenizer.strip(pred_caption)

                all_refs.append([tokenize(gt_captions[i])])
                all_hypos.append(tokenize(pred_caption))
            if num_batches is not None and len(all_hypos) >= num_batches * test_dataloader.batch_size:
                break
    
    scores = metrics.calculate(all_refs, all_hypos, train=False)

    print("\nEvaluation results:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    with open(model_folder / "evaluation_metrics.json", "w") as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the image captioning model using Natural Language Processing metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model folder.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model file (e.g., 'cptr_model.pth'). If not provided, defaults to 'cptr_model.pth'.")
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to evaluate (default: all).")
    args = parser.parse_args()
    evaluate(model_folder=args.model_path, model_name=args.model_name, num_batches=args.num_batches)