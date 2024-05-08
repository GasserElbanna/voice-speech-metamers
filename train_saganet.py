# This script finetunes Wav2Vec 2.0 on phoneme recognition.
# The Pytorch Lightning code skeleton is adapted from: https://github.com/nttcslab/byol-a

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

def main(config_path='finetune_config.yaml', layer_num=None) -> None:
    
    # Load config file
    finetuning_config = load_yaml_config(config_path)

    # Essentials
    logger = get_logger(__name__)
    logger.info(finetuning_config)
    seed_everything(finetuning_config.seed)

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
    
    # 5. Define Encoder (Wav2Vec2) model and a Decoder model

    #load pre-trained encoder model
    encoder = Encoder(finetuning_config.encoder.model_name,
                    finetuning_config.encoder.model_path,
                    finetuning_config.encoder.model_cache)
    
    select_layer = 'all' if layer_num is None else int(layer_num)
    print(select_layer)
    featurizer = Featurizer(finetuning_config.encoder.model_dim, 
                            finetuning_config.encoder.num_layers, 
                            select_layer)
    
    decoder = eval(finetuning_config.decoder.model)(
                finetuning_config.decoder.proj_dim,
                tokenizer.vocab_size,
                finetuning_config.decoder.hiddens,
                finetuning_config.decoder.activations,
                )

    logger.info('Loading models and dataloaders is done!')

    # 6. Define training parameters and callbacks

    #calculate the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / finetuning_config.trainer.gradient_accumulation_steps)
    total_batch_size = finetuning_config.dataloader.per_device_train_batch_size * finetuning_config.trainer.num_gpus * finetuning_config.trainer.gradient_accumulation_steps

    #calculate number of training epochs
    num_train_epochs = math.ceil(finetuning_config.trainer.max_train_steps / num_update_steps_per_epoch)

    #define name of the directory saving the checkpoints
    # name = (f"{finetuning_config.encoder.model_path.split('/')[-1]}"
    #         f'-finetune'
    #         f'-wordseparator'
    #         # f'-decoder{finetuning_config.decoder.proj_dim}'
    #         f'-layer{finetuning_config.encoder.select_layer}'
    #         f'-bs{finetuning_config.dataloader.per_device_train_batch_size}'
    #         f'-e{num_train_epochs}'
    #         f'-lr{finetuning_config.optimization.learning_rate}'
    #         f'-rs{finetuning_config.seed}'
    #         )
    name = (f"{finetuning_config.encoder.model_name}"
            f'_enclayer-{select_layer}'
            f'_data-{finetuning_config.data.data_name}'
            f'_decoder-{finetuning_config.decoder.model}'
            f'_bs-{finetuning_config.dataloader.per_device_train_batch_size}'
            f'_e-{num_train_epochs}'
            f'_lr-{finetuning_config.optimization.learning_rate}'
            f'_rs-{finetuning_config.seed}'
            )
    
    dir_name = f'{finetuning_config.callbacks.checkpoint_folder}/{name}'
    os.makedirs(dir_name, exist_ok=True)
    shutil.copy(config_path, dir_name)

    #define callbacks and logger
    model_checkpoint = ModelCheckpoint(dirpath=dir_name,
                                       filename='best{epoch:02d}-val_per{val_PER:.2f}', 
                                       monitor='val_PER',
                                       save_last=True,
                                       save_top_k=1, save_weights_only=False,
                                       auto_insert_metric_name=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(save_dir=finetuning_config.callbacks.checkpoint_folder, version=name, project=f"{finetuning_config.encoder.model_name}_{finetuning_config.decoder.model}_pr")
    # early_stop_callback = EarlyStopping(monitor="val_PER", min_delta=0.005, patience=10, verbose=False, mode="min")
    callbacks = [model_checkpoint, lr_monitor]
    if finetuning_config.callbacks.push_to_repo:
        local_dir = f'{dir_name}/huggingface_repo'
        os.makedirs(local_dir, exist_ok=True)
        hf_push_repo = HuggingFaceHubCallback(f'gelbanna/{name}', dir_name=dir_name, local_dir=local_dir, git_user='GasserElbanna')
        callbacks = [model_checkpoint, lr_monitor, hf_push_repo]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {finetuning_config.dataloader.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {finetuning_config.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {finetuning_config.trainer.max_train_steps}")

    # 7. Define Trainer and Learner
    trainer = pl.Trainer(accelerator='gpu',
                        devices=finetuning_config.trainer.num_gpus,
                        num_nodes=finetuning_config.trainer.num_nodes,
                        strategy=DDPStrategy(find_unused_parameters=False),
                        max_epochs=num_train_epochs,
                        callbacks=callbacks,
                        accumulate_grad_batches=finetuning_config.trainer.gradient_accumulation_steps,
                        logger=wandb_logger,
                        gradient_clip_val=finetuning_config.trainer.gradient_clip_val,
                        gradient_clip_algorithm='norm',
                        log_every_n_steps=finetuning_config.trainer.gradient_accumulation_steps,
                        val_check_interval=finetuning_config.trainer.check_val_steps,
                        # deterministic=True,
                        fast_dev_run=False,
                        num_sanity_val_steps=0,
                        profiler="simple")
    
    #run training
    ckpt_path=None

    global_step_offset = None
    if finetuning_config.trainer.resume or os.path.isfile(f'{dir_name}/last.ckpt'):
        ckpt_path = f'{dir_name}/last.ckpt'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        global_step_offset = checkpoint["global_step"]
        logger.info(f'Global Step Restored:{global_step_offset}')
    
    learner = Learner(config=finetuning_config, 
                    tokenizer=tokenizer,
                    encoder=encoder,
                    featurizer=featurizer,
                    decoder=decoder,
                    global_step=global_step_offset)

    trainer.fit(learner, train_dataloader, eval_dataloader, ckpt_path=ckpt_path)

    logger.info(f'Training is finished.')
    
if __name__ == '__main__':
    fire.Fire(main)