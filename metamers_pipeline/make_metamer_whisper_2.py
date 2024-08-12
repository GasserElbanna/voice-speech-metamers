import torch
import torchaudio
import numpy as np
import sys
import scipy
import librosa
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from transformers import AutoFeatureExtractor, AutoModel

torch.manual_seed(100)

# Initialize input_noise_init here
audio = '/om2/user/amagaro/voice-speech-metamers/metamers_pipeline/kell2018/metamers/psychophysics_wsj400_jsintest_inversion_loss_layer_RS0_I3000_N8/0_SOUND_million/orig.wav'

#sys.argv[2]
sr = 16000
signal, fs = torchaudio.load(audio)
if fs != sr:
    # make sure to resample to appropriate frequency
    print('resampling audio')
    resampler = T.Resample(fs, sr, dtype=signal.dtype)
    signal = resampler(signal)

if len(signal.shape)>1:
    # Reshape signal as necessary
    signal = torch.squeeze(signal)

# initialize random noise 
input_noise_init = torch.randn(signal.shape)
input_noise_init = input_noise_init * torch.std(signal) / torch.std(input_noise_init)
input_noise_init = torch.nn.parameter.Parameter(input_noise_init, requires_grad=True)

# initialize loss and optimizer
mse_loss = torch.nn.MSELoss()  # Assuming CrossEntropyLoss is being used

print('Initializing Optimizer')
iterations_adam = 30000
log_loss_every_num = 50
starting_learning_rate_adam = 0.1
adam_exponential_decay = 0.95

INIT_LR = 0.01
MAX_LR = 0.1
step_size = 2 * log_loss_every_num

# load in model 
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
whisper_encoder = AutoModel.from_pretrained("openai/whisper-base")#, cache_dir=cache_dir)
decoder_input_ids = torch.tensor([[1, 1]]) * whisper_encoder.config.decoder_start_token_id
whisper_encoder.eval()

print('Loaded in Whisper model')

# Get target embedding by running signal through model
with torch.no_grad():
    target = whisper_feature_extractor(signal.detach().cpu(), sampling_rate=sr, return_tensors="pt").input_features
    ## forward pass
    target = whisper_encoder(target, decoder_input_ids=decoder_input_ids)
    target = target.encoder_last_hidden_state.mean(1)
    print(target.shape)

## just get the features for the first time!
input_noise = whisper_feature_extractor(input_noise_init.detach().cpu(), sampling_rate=sr, return_tensors="pt").input_features
input_noise = input_noise.clone().requires_grad_()

optimizer = optim.SGD([input_noise], lr=INIT_LR)
clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=INIT_LR, max_lr=MAX_LR)

#import ipdb; ipdb.set_trace()

print('Performing optimization')
for i in range(iterations_adam + 1):
    optimizer.zero_grad()

    ## forward pass
    input_noise_encoded = whisper_encoder(input_noise, decoder_input_ids=decoder_input_ids)
    input_encoded_mean = input_noise_encoded.encoder_last_hidden_state.mean(1)
    # compute loss
    loss = mse_loss(input_encoded_mean, target)
    # backprop
    loss.backward()
    optimizer.step()
    clr.step()

    print(loss.item())


    # if i % log_loss_every_num == 0:
    #     input_noise_tensor_optimized = input_noise_init.detach().numpy()
    #     print(f'Saving Weights, {i/iterations_adam}%')
    #     np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)

    if i%50 == 0:
        print('Saving weights')
        #np.save('whisper/whisper_metamer.npy', input_noise_tensor_optimized)
        #scipy.io.wavfile.write('whisper/whisper_metamer.wav', sr, input_noise_tensor_optimized)
        signal_decode = librosa.feature.inverse.mel_to_audio(input_noise.detach().cpu().numpy(), sr=sr, n_fft=400, hop_length=160)
        np.save('whisper/whisper_2_metamer.npy', signal_decode)
        # torchaudio.save('whisper/whisper_metamer_2.wav', signal_decode, sample_rate=sr)

    # if i % log_loss_every_num == 0:
    #     loss_temp = loss_fn()
    #     print('Loss Value: ', loss_temp.item())
