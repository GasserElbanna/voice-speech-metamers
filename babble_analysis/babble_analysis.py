import scipy
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import tqdm

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
from learner_joint import LearnerJoint
from tokenizer import Tokenizer
from decoder import Speech_Decoder_Linear, Speaker_Decoder_Linear
from encoder import Speaker_Encoder, Speech_Encoder, Joint_Encoder


sr = 16000
min_snr, max_snr, snr_step = -10, 40, 5
# 15 min for 500 samples
n_samples = 500


#################################################  WHISPER  #################################################
# Load config file
config_path = '../config.yaml'
config = load_yaml_config(config_path)

#define a tokenizer for the vocabulary
tokenizer = Tokenizer(**config.text)

#load pre-trained encoder model
speech_encoder = Speech_Encoder(config.encoder.model_cache)

#define joint encoder
whisper = Joint_Encoder(config.saganet.d_model,
                        config.saganet.num_head,
                        config.saganet.dim_feedforward,
                        config.saganet.num_layers)

#define decoders
speech_decoder = Speech_Decoder_Linear()

checkpoint = "/om2/user/gelbanna/saganet/whisper_asr_bs-8_e-59_lr-0.0001_rs-42/best14-val_loss0.53.ckpt"
whisper = Learner.load_from_checkpoint(
                config=config, 
                checkpoint_path=checkpoint,
                tokenizer=tokenizer,
                joint_encoder=whisper,
                speech_encoder=speech_encoder,
                speech_decoder=speech_decoder)


# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_word = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper_word.config.forced_decoder_ids = None

def run_whisper(input):

    input_features = processor(input, sampling_rate=sr, return_tensors="pt", language='en').input_features
    # generate token ids
    predicted_ids = whisper_word.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


#################################################  SAGA-NET  #################################################

# load in joint 
config_path = "../config_joint.yaml"

# Load config file
config_joint = load_yaml_config(config_path)


#define a tokenizer for the vocabulary
tokenizer_joint = Tokenizer(**config.text)

speaker_encoder_joint = Speaker_Encoder(config_joint.encoder.model_cache)
speech_encoder_joint = Speech_Encoder(config_joint.encoder.model_cache)

#define joint encoder
saganet_joint = Joint_Encoder(config_joint.saganet.d_model,
                        config_joint.saganet.num_head,
                        config_joint.saganet.dim_feedforward,
                        config_joint.saganet.num_layers)

#define decoders
speech_decoder_joint = Speech_Decoder_Linear(d_model=config_joint.saganet.d_model)
speaker_decoder_joint = Speaker_Decoder_Linear()
checkpoint = "/om2/user/annesyab/SLP_Project_2024/saganet/saganet_d-704_atthead-8/best42-val_loss0.64.ckpt"
saganet = LearnerJoint.load_from_checkpoint(checkpoint_path=checkpoint,
                                                config=config_joint, 
                                                tokenizer=tokenizer_joint,
                                                speech_encoder=speech_encoder_joint,
                                                speaker_encoder=speaker_encoder_joint,
                                                joint_encoder=saganet_joint,
                                                speech_decoder=speech_decoder_joint,
                                                speaker_decoder = speaker_decoder_joint)

print('Loaded in joint model')


#################################################  HELPERS FUNCTIONS  #################################################
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

import editdistance as ed 
def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = list(p)#p.split(' ')
        t = list(t)#t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)

    return err / tot

#################################################  TEST STIMULI  #################################################


backgrounds = glob.glob('/om2/user/msaddler/spatial_audio_pipeline/assets/human_experiment_v00/background_cv08talkerbabble/*.wav')
signals = pd.read_csv('/om2/user/gelbanna/commonvoice_data_curated.csv').query('split == "test" and total_file_duration_in_s > 2')

results_joint = defaultdict(list)
results_whisper = defaultdict(list)
results_whisper_word = defaultdict(list)

def create_stim(row):


    # info = defaultdict(list)

    # load foreground
    foreground = load_signal(row['wav_path'])
    len_stim = foreground.shape[0]
    # info['foreground'].append(row['wav_path'])
    # info['foreground_idx'].append(row.index)

    # load and reshape background
    # info['background'].append(np.random.choice(backgrounds))
    # background = load_signal(info['background'][-1])
    background = load_signal(np.random.choice(backgrounds))
    background = np.concatenate([background]*(int(len_stim/3)+1))[:len_stim]

    # results_joint = defaultdict(list)
    # results_whisper = defaultdict(list)
    # gt = defaultdict(list)

    ground_truth = row['sentence'].replace(' ', '')

    # assert len(ground_truth) > 0, f'Ground truth is empty; {ground_truth}'

    for snr in range(min_snr, max_snr, snr_step):

        new_signal = combine_signal_and_noise(foreground, background, snr)

        # run sagenet

        joint_output = saganet({'input_values': torch.from_numpy(new_signal).unsqueeze(0)})[1][0].replace(' ', '')
        results_joint[snr].append(cer([joint_output], [ground_truth]))
        

        whisper_output = whisper({'input_values': torch.from_numpy(new_signal).unsqueeze(0)})[1][0].replace(' ', '')
        results_whisper[snr].append(cer([whisper_output], [ground_truth]))

        whisper_output = run_whisper(torch.from_numpy(new_signal))[0].replace(' ', '')
        results_whisper_word[snr].append(cer([whisper_output], [ground_truth]))

        
    joint_output = saganet({'input_values': torch.from_numpy(foreground).unsqueeze(0)})[1][0].replace(' ', '')
    results_joint[45].append(cer([joint_output], [ground_truth]))
    
    whisper_output = whisper({'input_values': torch.from_numpy(foreground).unsqueeze(0)})[1][0].replace(' ', '')
    results_whisper[45].append(cer([whisper_output], [ground_truth]))

    whisper_output = run_whisper(torch.from_numpy(foreground))[0].replace(' ', '')
    results_whisper_word[45].append(cer([whisper_output], [ground_truth]))



################################################# RUN MODEL TESTS + GET RESULTS  #################################################
    
tqdm.tqdm.pandas()
print(signals.shape)
signals = signals.sample(n=n_samples)
signals.progress_apply(create_stim, axis = 1)



snrs = []
joint_results = []
joint_sem = []
whisper_results = []
whisper_sem = []
whisper_word_results = []
whisper_word_sem = []

for snr in results_joint.keys():
    snrs.append(snr)
    joint_results.append(np.mean(results_joint[snr]))
    joint_sem.append(scipy.stats.sem(results_joint[snr]))

    whisper_results.append(np.mean(results_whisper[snr]))
    whisper_sem.append(scipy.stats.sem(results_whisper[snr]))

    whisper_word_results.append(np.mean(results_whisper_word[snr]))
    whisper_word_sem.append(scipy.stats.sem(results_whisper_word[snr]))

    
#################################################  PLOT RESULTS  #################################################
    
import matplotlib.pyplot as plt

plt.figure()
plt.errorbar(x = snrs, y = joint_results, yerr = joint_sem, color = 'purple')
plt.errorbar(x = snrs, y = whisper_results, yerr = whisper_sem, color = 'teal')
plt.errorbar(x = snrs, y = whisper_word_results, yerr = whisper_word_sem, color = 'lightblue')
plt.title('Character Error Rate by SNR')
plt.legend(['SAGA-NET', 'Whisper', 'Whisper_word'])
plt.ylabel('Characeter Error Rate (CER)')
# plt.ylim(0, 1)
plt.xlabel('SNR (dB)')
plt.savefig('cer_by_snr.png')


breakpoint()