import sys
from datasets import load_dataset

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import aiohttp

sys.path.append("..")

import config

class DatasetLoader:
    def __init__(self, dataset_type=config.DATASET, 
                img_height=config.IMG_HEIGHT, 
                img_width=config.IMG_WIDTH, 
                batch_size_train=config.BATCH_SIZE_TRAIN,
                batch_size_val=config.BATCH_SIZE_VAL,
                batch_size_test=config.BATCH_SIZE_TEST,
                split_ratio=config.SPLIT_RATIO,
                shuffle_train=True,
                shuffle_val=False,
                shuffle_test=False,
                seed=config.RANDOM_SEED):
        self.dataset_type = dataset_type
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.split_ratio = split_ratio
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.seed = seed
        
        self.tf_compose = transforms.Compose(
            [
                transforms.Resize(size=[img_height, img_width], interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    
    # @staticmethod
    def _transforms(self, examples):
        examples["pixel_values"] = [self.tf_compose(image.convert("RGB")) for image in examples["pixel_values"]]
        examples["description"] = [desc[0].strip() if isinstance(desc, list) else desc.strip() for desc in examples["description"]]
        return examples
    
    # TODO?
    def _transforms_coco(self, examples):
        # leave only one caption per image for simplicity and strip of trailing spaces/newlines
        examples["description"] = [desc[0].strip() for desc in examples["description"]]
        return examples

    def load_data(self):
        # Placeholder for loading data logic        
        if self.dataset_type == config.Dataset.DOCCI:
            dataset = self._load_docci()
        elif self.dataset_type == config.Dataset.DOCCI_IIW:
            dataset = self._load_docci_iiw()
        elif self.dataset_type == config.Dataset.COCO:
            dataset = self._load_coco()
        elif self.dataset_type == config.Dataset.FLICKR:
            dataset = self._load_flickr()
        else:
            raise ValueError("Unsupported dataset type")

        dataset = dataset.rename_column("image", "pixel_values")

        self.dataset_train = dataset['train']
        self.dataset_train.set_transform(self._transforms)

        if 'val' in dataset:
            self.dataset_val = dataset['val']
            self.dataset_val.set_transform(self._transforms)

        self.dataset_test = dataset['test']
        self.dataset_test.set_transform(self._transforms)

        self.train_dataloader = DataLoader(dataset=self.dataset_train, 
                              batch_size=self.batch_size_train, # how many samples per batch?
                              num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=self.shuffle_train) # shuffle the data?

        if self.dataset_val is not None:
            self.val_dataloader = DataLoader(dataset=self.dataset_val, 
                                batch_size=self.batch_size_val, 
                                num_workers=0, 
                                shuffle=self.shuffle_val)

        self.test_dataloader = DataLoader(dataset=self.dataset_test, 
                             batch_size=self.batch_size_test, 
                             num_workers=0, 
                             shuffle=self.shuffle_test) # don't usually need to shuffle testing data

    def _load_docci(self):
        print("Loading DOCCI dataset...")
        # Prolonged timeout for loading the Google dataset because they are losers who don't host on HuggingFace: https://github.com/huggingface/datasets/issues/7164#issuecomment-2439589751

        dataset = load_dataset('google/docci', name='docci', trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=10000)}})
        dataset = dataset.remove_columns(["example_id"])
        print(dataset)
        return dataset
    
    def _load_docci_iiw(self):
        print("Loading IIW dataset...")
        dataset = self._load_docci()
        return None  # Placeholder
    
    def _load_coco(self):
        print("Loading COCO dataset...")

        dataset = load_dataset("lmms-lab/COCO-Caption")
        dataset = dataset.remove_columns(['question_id', 'question', 'id', 'license', 'file_name', 'coco_url', 'date_captured', 'height', 'width'])
        dataset = dataset.rename_column("answer", "description")
        
        # split dataset['val] into train and test as 80/20
        dataset = dataset['val'].train_test_split(test_size=self.split_ratio, seed=self.seed)

        return dataset
    
    def _load_flickr(self):
        print("Loading Flickr30k dataset...")

        dataset = load_dataset("nlphuji/flickr30k")
        dataset = dataset.remove_columns(['sentids', 'split', 'img_id', 'filename'])
        dataset = dataset.rename_column("caption", "description")
        print(dataset)
        
        # split dataset['test'] into train and test as 80/20
        dataset = dataset['test'].train_test_split(test_size=self.split_ratio, seed=self.seed)
                
        return dataset
    
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader
    
    # takes special tokens into account
    def get_max_description_length_in_tokens_train(self, tokenizer, max_sequence_length):
        max_length = 0
        for batch in self.train_dataloader:
            samples = batch['description']
            for sample in samples:
                tokenized = tokenizer.encode(sample, max_seq_length=max_sequence_length, verbose=False)['input_ids']
                desc_length = len(tokenized)
                if desc_length > max_length:
                    max_length = desc_length
        return max_length
    
    def get_max_description_length_in_tokens_test(self, tokenizer, max_sequence_length):
        max_length = 0
        for batch in self.test_dataloader:
            samples = batch['description']
            for sample in samples:
                tokenized = tokenizer.encode(sample, max_seq_length=max_sequence_length, verbose=False)['input_ids']
                desc_length = len(tokenized)
                if desc_length > max_length:
                    max_length = desc_length
        return max_length

    def get_max_description_length_in_tokens(self, tokenizer, max_sequence_length, descriptions=None):
        if descriptions is not None:
            max_length = self.get_max_description_length_in_tokens_train(tokenizer, max_sequence_length)
            test_max_length = self.get_max_description_length_in_tokens_test(tokenizer, max_sequence_length)
            if test_max_length > max_length:
                max_length = test_max_length
        else:
            max_length = 0
            for desc in descriptions:
                tokenized = tokenizer.encode(desc, max_seq_length=max_sequence_length, verbose=False)['input_ids']
                desc_length = len(tokenized)
                if desc_length > max_length:
                    max_length = desc_length
        return max_length

    # length is measured in characters
    def get_max_description_length_train(self):
        max_length = 0
        for batch in self.train_dataloader:
            samples = batch['description']
            for sample in samples:
                desc_length = len(sample)
                if desc_length > max_length:
                    max_length = desc_length
        return max_length

    # length in measured in characters
    def get_max_description_length_test(self):
        max_length = 0
        for batch in self.test_dataloader:
            samples = batch['description']
            for sample in samples:
                desc_length = len(sample)
                if desc_length > max_length:
                    max_length = desc_length
        return max_length
    
    # doesn't consider val set
    def get_max_description_length(self):
        max_length = self.get_max_description_length_train()
        test_max_length = self.get_max_description_length_test()
        if test_max_length > max_length:
            max_length = test_max_length
        return max_length

if __name__ == "__main__":
    loader = DatasetLoader()
    loader.load_data()
    train_loader = loader.get_train_dataloader()
    test_loader = loader.get_test_dataloader()
    
    for batch in train_loader:
        print(batch)
        break
    
    for batch in test_loader:
        print(batch)
        print(len(batch['pixel_values']))
        break
        