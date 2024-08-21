import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel
from speechbrain.inference.speaker import EncoderClassifier

class Speech_Encoder(torch.nn.Module):
    def __init__(self, cache_dir, sampling_rate=16000):
        super(Speech_Encoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        self.whisper_encoder = AutoModel.from_pretrained("openai/whisper-base", cache_dir=cache_dir)
        self.decoder_input_ids = torch.tensor([[1, 1]]) * self.whisper_encoder.config.decoder_start_token_id
        #self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

    def _preprocess(self, input_):
        if len(input_.shape)>1:
            input_ = torch.squeeze(input_)
        input_feat = self.whisper_feature_extractor(input_.detach().cpu(), sampling_rate=self.sampling_rate, return_tensors="pt")
        return input_feat.input_features
    
    def forward(self, input_values):
        # Forward pass for whisper branch
        self.whisper_encoder.eval()
        input_values = [self._preprocess(input_) for input_ in input_values]
        output_values = [self.whisper_encoder(input_.to("cuda"), decoder_input_ids=self.decoder_input_ids.to("cuda")) for input_ in input_values]
        output_values = [output_.encoder_last_hidden_state for output_ in output_values]
        return torch.stack(output_values,dim=0).squeeze(1)

class Speaker_Encoder(torch.nn.Module):
    def __init__(self, cache_dir):
        super(Speaker_Encoder, self).__init__()
        self.ecapa_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=cache_dir)
        self.ecapa_encoder.device = "cuda"

    def forward(self, input_values):
        # Forward pass for ECAPA branch
        self.ecapa_encoder.eval()
        speaker_embedding = [self.ecapa_encoder.encode_batch(input_.squeeze(0).to("cuda")) for input_ in input_values] # shape batch x 1 x embeddings
        return torch.stack(speaker_embedding,dim=0).squeeze(1)
    
class Joint_Encoder(torch.nn.Module):
    def __init__(self, d_model=704, num_head=8, dim_feedforward=512, num_layers=2):
        super(Joint_Encoder, self).__init__()
        self.transformer_encoder_single = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head, dim_feedforward=dim_feedforward, batch_first=True)
        self.joint_encoder = nn.TransformerEncoder(self.transformer_encoder_single, num_layers=num_layers)

    def forward(self, input_values):
        # Forward pass for joint branch
        joint_embedding = self.joint_encoder(input_values) # shape batch x timeframes x embeddings
        return joint_embedding


if __name__ == '__main__':
    speech_model = Speech_Encoder("../cache_data")
    speaker_model = Speaker_Encoder("../cache_data")
    joint_model = Joint_Encoder()
    x = [torch.ones(50000), torch.ones(35000)]
    #y1 = speech_model(x)
    x2 = [torch.ones(1,1,50000), torch.ones(1,1,35000)]
    y1 = speech_model(x2)
    y2 = speaker_model(x2)
    x = torch.ones((1,1500,704))
    y3 = joint_model(x)
    # print("Speech Embeddings", len(y1), y1[0].shape)
    # print("Speaker Embeddings", len(y2), y2[0].shape)
    print("Speech Embeddings", len(y1), y1.shape)
    print("Speaker Embeddings", len(y2), y2.shape)
    print(y3.shape)