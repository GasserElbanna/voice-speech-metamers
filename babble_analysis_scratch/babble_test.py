import scipy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.append('/om2/user/salavill/misc/voice-speech-metamers/')
from utils import *
from learner import Learner
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder


sr = 16000
min_snr, max_snr, snr_step = 0, 40, 5

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper.config.forced_decoder_ids = None

def run_whisper(input):

    input_features = processor(input, sampling_rate=sr, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = whisper.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


# load in joint 
config_path = "../config.yaml"

# Load config file
config = load_yaml_config(config_path)


#define a tokenizer for the vocabulary
tokenizer = Tokenizer(**config.text)

speaker_encoder = Speaker_Encoder(config.encoder.model_cache)
speech_encoder = Speech_Encoder(config.encoder.model_cache)

#define joint encoder
saganet = Joint_Encoder(config.saganet.d_model,
                        config.saganet.num_head,
                        config.saganet.dim_feedforward,
                        config.saganet.num_layers)

#define decoders
speech_decoder = Speech_Decoder_Linear()
speaker_decoder = Speaker_Decoder_Linear()


checkpoint = "/om2/user/annesyab/SLP_Project_2024/saganet/saganet_d-704_atthead-8/best42-val_loss0.64.ckpt"
saganet = Learner.load_from_checkpoint(checkpoint_path=checkpoint,
                                                config=config, 
                                                tokenizer=tokenizer,
                                                speech_encoder=speech_encoder,
                                                speaker_encoder=speaker_encoder,
                                                joint_encoder=saganet,
                                                speech_decoder=speech_decoder,
                                                speaker_decoder = speaker_decoder,)

print('Loaded in joint model')



def combine_signal_and_noise(signal, noise, snr, mean_subtract=True):
    '''
    Adds noise to signal with the specified signal-to-noise ratio (snr).
    If snr is finite, the noise waveform is rescaled and added to the
    signal waveform. If snr is positive infinity, returned waveform is
    equal to the signal waveform. If snr is negative inifinity, returned
    waveform is equal to the noise waveform.
    
    Args
    ----
    signal (np.ndarray): signal waveform
    noise (np.ndarray): noise waveform
    snr (float): signal-to-noise ratio in dB
    mean_subtract (bool): if True, signal and noise are first de-meaned
        (mean_subtract=True is important for accurate snr computation)
    
    Returns
    -------
    signal_and_noise (np.ndarray) signal in noise waveform
    '''
    rms = lambda stim: np.sqrt(np.mean(stim * stim))

    if mean_subtract:
        signal = signal - np.mean(signal)
        noise = noise - np.mean(noise)        
    if np.isinf(snr) and snr > 0:
        signal_and_noise = signal
    elif np.isinf(snr) and snr < 0:
        signal_and_noise = noise
    else:
        rms_noise_scaling = rms(signal) / (rms(noise) * np.power(10, snr / 20))
        signal_and_noise = signal + rms_noise_scaling * noise
    return signal_and_noise

def load_signal(file):
    signal, fs = torchaudio.load(file)
    if fs != sr:
        # make sure to resample to appropriate frequency
        print('resampling audio')
        resampler = T.Resample(fs, sr, dtype=signal.dtype)
        signal = resampler(signal)

    if len(signal.shape)>1:
        # Reshape signal as necessary
        signal = torch.squeeze(signal)

    return signal.numpy()



backgrounds = glob.glob('/om2/user/msaddler/spatial_audio_pipeline/assets/human_experiment_v00/background_cv08talkerbabble/*.wav')
signals = pd.read_csv('/om2/user/gelbanna/commonvoice_data_curated.csv').query('split == "test" and total_file_duration_in_s > 2')

def create_stim(row):


    info = defaultdict(list)

    # load foreground
    foreground = load_signal(row['wav_path'])
    len_stim = foreground.shape[0]
    info['foreground'].append(row['wav_path'])
    info['foreground_idx'].append(row.index)
    info['gt'].append(row['sentence'])

    # load and reshape background
    info['background'].append(np.random.choice(backgrounds))
    background = load_signal(info['background'][-1])
    background = np.concatenate([background]*(int(len_stim/3)+1))[:len_stim]

    output = defaultdict(list)
    info['output'].append(output)
    
    output['snr'].append(45)
    # run clean signal through the models
    output['saganet'].append(saganet({'input_values': torch.from_numpy(foreground).unsqueeze(0)})[1])

    output['whisper'].append(run_whisper(torch.from_numpy(foreground)))


    for snr in range(min_snr, max_snr, snr_step):

        output['snr'].append(snr)

        new_signal = combine_signal_and_noise(foreground, background, snr)

        # run sagenet
        output['saganet'].append(saganet({'input_values': torch.from_numpy(new_signal).unsqueeze(0)})[1])

        output['whisper'].append(run_whisper(torch.from_numpy(new_signal)))

        breakpoint()


signals.apply(create_stim, axis = 1)

