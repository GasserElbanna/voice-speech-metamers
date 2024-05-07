from speechbrain.inference.speaker import EncoderClassifier
import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WhisperModel

class Speech_Encoder_Transformer(torch.nn.Module):
    def __init__(self, sampling_rate=16000,d_model=704,num_head=1,dim_feedforward=512,num_layers=6):
        super(Speech_Encoder_Transformer, self).__init__()
        self.sampling_rate = sampling_rate
        self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-base")
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.whisper_encoder.config.decoder_start_token_id
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        self.ecapa_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

        # Define downsampling layers
        # No downsampling needed as output of ECAPA-TDNN pooled and concatenating with the word embedding 


        # Define Transformer layers
        self.transformer_encoder_single = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=dim_feedforward, batch_first=True)
    
        self.transformer_encoder_stack = nn.TransformerEncoder(self.transformer_encoder_single, num_layers=num_layers)

        # Define prediction heads
        self.next_word_prediction_head = nn.Linear(128,self.word_vocab)
        self.speaker_recognition_head = nn.Linear(128,self.num_speaker_class)
        self.speaker_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Forward pass for whisper branch
        x_whisper = self.whisper_feature_extractor(x, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        whisper_embedding = self.whisper_encoder(x_whisper, decoder_input_ids=self.decoder_input_ids).encoder_last_hidden_state # shape batch x timeframes x embeddings

        # Forward pass for ECAPA branch
        ecapa_embedding = self.ecapa_encoder.encode_batch(x) # shape batch x 1 x embeddings
        
        # Concatenate downscaled embeddings
        # concatenate the word embeddings with speaker embeddings without changing the frame rate 
        batch, feature, embeddings = whisper_embedding.shape
        ecapa_embedding = ecapa_embedding.expand(-1, feature, -1)
        concatenated_embeddings = torch.cat((whisper_embedding, ecapa_embedding), dim=-1)

        # Transformer layers
        transformer_output = self.transformer_encoder_stack(concatenated_embeddings)

        # Task-specific heads
        # word_prediction = self.next_word_prediction_head(transformer_decoder_output)
        # speaker_recognition = self.speaker_recognition_head(transformer_output)
        # speaker_recognition = self.speaker_softmax(speaker_recognition)

        #next_word_prediction = self.next_word_prediction_head(transformer_output

        return transformer_output


if __name__ == '__main__':
    model = Speech_Encoder_Transformer()
    # x = {}
    x = [torch.ones((2,35000))]
    print(x)
    # compute the forward pass
    y = model(x)
    # print(y.keys(), len(y["encoder_features_0"]), y["encoder_features_0"][1].shape)
    print(len(y))
    print(len(y[0]), len(y[1]))
    print(y[0][0].shape, y[1][0].shape)