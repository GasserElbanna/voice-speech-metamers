from speechbrain.inference.speaker import EncoderClassifier
import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoFeatureExtractor, WhisperModel

class Speech_Encoder(torch.nn.Module):
    def __init__(self, sampling_rate=16000):
        super(Speech_Encoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-base")
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.whisper_encoder.config.decoder_start_token_id
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base") 

    def forward(self, x):
        # Forward pass for whisper branch
        x = self.whisper_feature_extractor(x, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        speech_embedding = self.whisper_encoder(x, decoder_input_ids=self.decoder_input_ids).encoder_last_hidden_state # shape batch x timeframes x embeddings

        return speech_embedding
    

class Speaker_Encoder(torch.nn.Module):
    def __init__(self):
        super(Speaker_Encoder, self).__init__()
        self.ecapa_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def forward(self, x):
        # Forward pass for ECAPA branch
        speaker_embedding = self.ecapa_encoder.encode_batch(x) # shape batch x 1 x embeddings

        return speaker_embedding
    
class Joint_Encoder(torch.nn.Module):
    def __init__(self,d_model=704,num_head=1,dim_feedforward=512,num_layers=6):
        super(Speaker_Encoder, self).__init__()
        self.transformer_encoder_single = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=dim_feedforward, batch_first=True)
        self.joint_encoder = nn.TransformerEncoder(self.transformer_encoder_single, num_layers=num_layers)

    def forward(self, x):
        # Forward pass for joint branch
        joint_embedding = self.joint_encoder(x) # shape batch x timeframes x embeddings

        return joint_embedding


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