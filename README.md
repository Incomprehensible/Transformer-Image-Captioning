# Transformer-Image-Captioning
Final Project for Machine Learning course (700.851) at the University of Klagenfurt.
 
The project focuses on implementation and training of a Transformer-based image captioning model. It explores the performance of various encoder-decoder configurations, specifically comparing traditional CNN backbones against modern Vision Transformer (ViT) approaches, all integrated into an architecture inspired by the CPTR (`CPTR: Full Transformer Network for Image Captioning`) [architecture paper](https://arxiv.org/abs/2101.10804).

This repository contains:
* A comprehensive framework for training and evaluating Transformer-based image captioning models. 
* Captioning models implementation.

---

## Project Description

The primary goal of this project is to investigate how different visual encoding strategies affect the quality and semantic accuracy of generated captions. By treating image features as sequential inputs to a Transformer decoder, we bridge the gap between spatial feature extraction and the global attention mechanisms of Transformers.
The project also highlights the influence of the evaluation metrics on caption quality tuning.

---

## Model Architecture Details

### Encoder Architectures
The framework supports 4 primary encoder configurations:

* **CNN Encoder**: Utilizes a CNN backbone to extract spatial feature maps. Pretrained and imported as `resnet50`.
* **Vision Transformer (ViT)**: Processes images as a sequence of fixed-size flattened patches and applies self-attention to capture global context. It includes CLS token and patch tokens. Pretrained and imported as `vit-base-patch16-224-in21k` or `vit-large-patch16-224-in21k`.
* **Custom CPTR Style**: Based on the CPTR paper, which flattens a sequence of fix-sized embedded patches and applies learnable positional embeddings. Trained from scratch.
* **CNN-CPTR**: A hybrid approach that combines CNN feature extraction with CPTR-style patch embedding and positional encoding. CNN pretrained, CPTR trained from scratch.

ViT-style encoder supports 3 configuration options:

| Strategy    | Description                     | Output Shape |
|-------------|----------------------------------|--------------|
| `CLS_TOKEN`   | Use only the CLS token            | `[B, 1, D]`  |
| `PATCHES`     | Use patch tokens only             | `[B, N, D]`  |
| `HYBRID`      | CLS + patch tokens                | `[B, N+1, D]`|

### Tokenizers
This project utilizes two types of tokenizers for text processing:
* **Byte-Pair Encoding (BPE)**: Custom implementation of BPE tokenizer.
* **GPT-2 Tokenizer**: A widely used tokenizer that supports a large vocabulary. It is pretrained and imported from Hugging Face's `transformers` library.

We added support for standard pre-trained tokenizers. The project also includes a pipeline to train custom tokenizers on a specific image-caption corpus for optimized vocabulary coverage.

### Datasets Supported
* **Flickr30k**: Medium-sized dataset with challenging captions rich in detail and grammar. Can contain hidden context in text.
* **COCO**: Largest dataset with more concise captions and less complex grammar. It features more straightfoward situations and objects.
* **DOCCI**: Smallest dataset. Focused on long-form, highly descriptive image captioning.

Datasets stats can be found in [`tokenizer_data`](./tokenizer_data) folder.

---

## Evaluation Metrics

The project implements a multi-metric evaluation suite to assess different aspects of caption quality:

* **BLEU (1-4)**: Evaluates n-gram overlap between predicted and reference captions.
* **METEOR**: Focuses on semantic alignment, accounting for stem matching and synonyms via WordNet.
* **GLEU**: Provides a stable sentence-level evaluation score.
* **Cross-Entropy Loss**: Used for optimization and monitoring mathematical convergence.

---

## Key Features

* **Flexible Configuration**: Every aspect of the project, from architecture layers to learning rates, is defined in a centralized, readable configuration file (`config.py`).
* **Advanced Training Pipeline**:
    * **Schedulers**: Support for OneCycleLR and CosineAnnealing for stable convergence.
    * **Early Stopping**: Logic that monitors validation performance (loss or metrics) to prevent overfitting.
    * **Real-Time Metrics**: Evaluation of BLEU-4 and METEOR scores during the training loop.
* **Smart Result Export**: The pipeline automatically identifies and exports the best model states based on the lowest validation loss or highest linguistic metrics.
* **Evaluation Scripts**: Dedicated scripts for tokenizer, model architecture testing, single-image inference, and metric calculation.

---

## Technical Instructions

### Project Configuration
The project is configured through a single [`config.py`](./config.py) file, which defines all parameters for model. Adjust the parameters in this file to set up your training and evaluation runs. This includes selecting the encoder architecture, tokenizer type, dataset, training hyperparameters.
However, for running the inference on pretrained models, you don't need to modify the `config.py` file. Instead, the scripts will load the pretrained model and its corresponding configuration from the specified folder in the code.

### Installation and Build
Our code is compatible with Python 3.10 (We used Python 3.10.12).
1. Clone the repository:
   ```bash
   git clone https://github.com/Incomprehensible/Transformer-Image-Captioning.git
   cd Transformer-Image-Captioning
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up git-LFS for handling large model files:
   ```bash
   sudo apt-get install git-lfs
   git lfs install
   git lfs fetch
   git lfs checkout
   ```
4. Run inference by executing the [`run_inference.ipynb`](./run_inference.ipynb) notebook. Make sure to specify the path to the pretrained model folder.
5. Train a new model by editing the [`config.py`](./config.py) file with your desired settings and running the [`train_model.ipynb`](./train_model.ipynb) notebook.
6. Evaluate the model using the [`evaluate_metrics.py`](./evaluate_metrics.py) script:
   ```bash
    python evaluate_metrics.py --model_path <path_to_your_model_folder> # --model_name <model_file_name> --num_batches 10 (optional, specify number of batches to evaluate)
    ```
    For example:
    ```bash
    python evaluate_metrics.py --num_batches 10 --model_path experiments/config_20260129-231542 # defaults to using 'cptr_model.pth' as the model file name
    ```
7. Visualize self-attention maps using the [`visualize_self_attention.ipynb`](./visualization/visualize_self_attention.ipynb) notebook. Make sure to specify the path to the pretrained model folder.
8. Visualize the model architecture using the [`visualize_architecture.ipynb`](./visualization/visualize_architecture.ipynb) notebook. Make sure to specify the path to the pretrained model folder.
9. Train a custom tokenizer using the [`train_bpe_tokenizer.ipynb`](./train_bpe_tokenizer.ipynb) notebook. Adjust the dataset and training parameters as needed.

## Key Insights
* We found that the choice of visual encoder significantly impacts the quality of generated captions.
* More complex architectures with more parameters do not always yield better results, especially when trained on smaller datasets like Flickr and COCO.
* ViT encoders tend to overfit on smaller datasets, while CNN-based encoders provide more stable performance overall.
* Evaluation metrics can sometimes be misleading; optimizing for test loss or any other single metric may not always correlate with human judgment of caption quality, highlighting the importance of using multiple metrics for a comprehensive evaluation.
* Longest trained model may result in better captions than a test loss-optimized model if judged by a human.
* All evaluation metrics used are rather naive and do not capture the semantic quality of the generated captions ideally. Future work should explore more sophisticated evaluation methods that better align with human judgment (e.g. CIDEr).
* Training a model based on custom CPTR-style encoder without pretraining didn't yield good results. Training both encoder and decoder from scratch in parallel may be too much for the model to handle.
* Using a larger tokenizer (GPT-2) doesn't necessarily lead to better performance, especially when the dataset is small. It mostly leads to longer convergence.
* Training a model to produce very long captions based on DOCCI dataset is very challenging. The model struggles to learn the long-range dependencies required for generating coherent long captions, and the evaluation metrics used (BLEU, METEOR) are not well-suited for assessing the quality of long captions. We noticed that larger models trained on DOCCI show superior command of language and grammar but may struggle to see concepts. With time they start to produce captions degrading in quality, or start repeating words and phrases. The model may also hallucinate, so it's crucial to choose a suitable encoder and training strategy for this dataset.

## Warnings
* Note that we tried to preserve as many checkpoints as possible (and kinda overdid it) for the purpose of course project validation, therefore the cloning of the repository may take a long time and require a lot of disk space.
* Technically `vit-large-patch16-224-in21k` is supported in the code but we never tested it.
* Ensure that you have a stable internet connection when running the code.