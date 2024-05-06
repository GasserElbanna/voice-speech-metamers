from speechbrain.inference.speaker import EncoderClassifier
import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WhisperModel

class Speech_Model_Transformer(torch.nn.Module):
    def __init__(self):
        super(Speech_Model_Transformer, self).__init__()
        self.sampling_rate = 16000
        self.num_speaker_class = 578 # change this after analyzing dataset
        self.word_vocab = 2610 # change this after analyzing dataset -> len(vocab_dict)
        self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-base")
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.whisper_encoder.config.decoder_start_token_id
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        self.ecapa_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

        # Define downsampling layers
        self.whisper_downsample = nn.Conv1d(in_channels=512, out_channels=64, kernel_size=1)
        self.ecapa_downsample = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=1)

        # Define Transformer layers
        self.transformer_encoder_single = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer_decoder_single = nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer_encoder_stack = nn.TransformerEncoder(self.transformer_encoder_single, num_layers=6)
        self.transformer_decoder_stack = nn.TransformerDecoder(self.transformer_decoder_single, num_layers=6)

        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.whisper_encoder.config.decoder_start_token_id

        # Define prediction heads
        self.next_word_prediction_head = nn.Linear(128,self.word_vocab)
        self.speaker_recognition_head = nn.Linear(128,self.num_speaker_class)
        self.speaker_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Forward pass for whisper branch
        x_whisper = self.whisper_feature_extractor(x, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        whisper_embedding = self.whisper_encoder(x_whisper, decoder_input_ids=self.decoder_input_ids).encoder_last_hidden_state
        whisper_embedding = whisper_embedding.permute(0, 2, 1)
        downsampled_whisper_embedding = self.whisper_downsample(whisper_embedding)
        downsampled_whisper_embedding = downsampled_whisper_embedding.permute(0, 2, 1)
        downsampled_whisper_embedding = torch.mean(downsampled_whisper_embedding, dim=1)

        # Forward pass for ECAPA branch
        ecapa_embedding = self.ecapa_encoder.encode_batch(x)
        ecapa_embedding = ecapa_embedding.permute(0, 2, 1)
        downsampled_ecapa_embedding = self.ecapa_downsample(ecapa_embedding)
        downsampled_ecapa_embedding = downsampled_ecapa_embedding.permute(0, 2, 1)
        downsampled_ecapa_embedding = torch.squeeze(downsampled_ecapa_embedding,dim=1)

        # Concatenate downscaled embeddings
        concatenated_embeddings = torch.cat((downsampled_whisper_embedding, downsampled_ecapa_embedding), dim=-1)

        # Transformer layers
        transformer_output = self.transformer_encoder_stack(concatenated_embeddings)
        #transformer_decoder_output = self.transformer_decoder_stack(self.decoder_input_ids, transformer_output)
        transformer_decoder_output = self.transformer_decoder_stack(concatenated_embeddings, transformer_output)

        # Task-specific heads
        word_prediction = self.next_word_prediction_head(transformer_decoder_output)
        speaker_recognition = self.speaker_recognition_head(transformer_output)
        speaker_recognition = self.speaker_softmax(speaker_recognition)

        #next_word_prediction = self.next_word_prediction_head(transformer_output

        return word_prediction, speaker_recognition

# Define model
model = Speech_Model_Transformer()
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_data = torch.tensor(ds[0]["audio"]["array"])
output_word, output_speaker = model.forward(input_data)
print(output_word.shape)
print(output_speaker.shape)