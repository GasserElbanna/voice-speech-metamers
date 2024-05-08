import os
import ast
import fire
import math
import shutil
from glob import glob

from utils import *
from encoder import Encoder
from learner import Learner
from data import DataCollator
from tokenizer import Tokenizer
from featurizer import Featurizer
from decoders.mlp import FrameLevelLinear
from decoders.lstm import LSTM

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from hf_hub_lightning import HuggingFaceHubCallback

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from transformers import Wav2Vec2FeatureExtractor

torch.set_float32_matmul_precision('medium' or 'high')

# 1. Read data and split it into train and test

#read csv data as dataframe
#data_name = '*' if finetuning_config.data.data_name == 'all' else finetuning_config.data.data_name
#data_files = glob(f'{finetuning_config.data.data_dir}/{data_name}_*.csv')

data_train = '/om2/user/msaddler/spatial_audio_pipeline/assets/commonvoice_9_en/manifest_core_train.pdpkl'
df_train = pd.read_pickle(data_train)

data_valid = '/om2/user/msaddler/spatial_audio_pipeline/assets/commonvoice_9_en/manifest_core_valid.pdpkl'
df_valid = pd.read_pickle(data_valid)

df['speaker_id'] = df.speaker_id.astype(str)
#convert string representation of list into a list
df['phone'] = df.apply(lambda x: ast.literal_eval(x.phone), axis=1)

#split data into train and test
# train_df = df.loc[df.split.str.contains('train')]
# test_df = df.loc[df.split.str.contains('dev')]

#convert dataframes into huggingface dataset
raw_datasets = DatasetDict()
raw_datasets['train'] = Dataset.from_pandas(df_train)
raw_datasets["validation"] = Dataset.from_pandas(df_valid)

# #load audio files
# raw_datasets = raw_datasets.map(load_audios)

# 2. Define Tokenizer and Feature Extractor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

#define a tokenizer for the vocabulary
tokenizer = processor.tokenizer()
#define a feature extractor for the model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", 
                                                            cache_dir=finetuning_config.cache_dir)

# 3. Preprocess and tokenize data

# load via mapped files via path
cache_file_names = None
if finetuning_config.data.train_cache_file_path is not None:
    cache_file_names = {"train": f'{finetuning_config.data.train_cache_file_path}/finetune_train_{finetuning_config.data.data_name}', 
                        "validation": f'{finetuning_config.data.validation_cache_file_path}/finetune_dev_{finetuning_config.data.data_name}'}
remove_columns = raw_datasets['train'].column_names
remove_columns.remove('text')
vectorized_datasets = raw_datasets.map(prepare_data,
                                    remove_columns=remove_columns,
                                    cache_file_names=cache_file_names,
                                    fn_kwargs={"feat_ext":feature_extractor, "tokenizer":tokenizer, "sr": finetuning_config.data.sampling_rate})

# 4. Define DataCollator and DataLoaders
data_collator = DataCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)
train_dataloader = DataLoader(
    vectorized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=finetuning_config.dataloader.per_device_train_batch_size,
    num_workers=finetuning_config.dataloader.num_workers,
)
eval_dataloader = DataLoader(
    vectorized_datasets["validation"], 
    collate_fn=data_collator, 
    batch_size=finetuning_config.dataloader.per_device_eval_batch_size,
    num_workers=finetuning_config.dataloader.num_workers,
)