import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

class Speech_Decoder_Linear(torch.nn.Module):
    def __init__(self, word_vocab=30, d_model=704, hidden_size=512):
        super(Speech_Decoder_Linear, self).__init__()
        self.word_vocab = word_vocab # change this after analyzing dataset -> len(vocab_dict)
        self.linear_project_word = nn.Linear(d_model, hidden_size)
        self.linear_classifier_word = nn.Linear(hidden_size, self.word_vocab)
        self.act = nn.ReLU()

    def forward(self, x):
        # Forward decoder pass on word branch
        word_embedding_decoded = self.act(self.linear_project_word(x))
        word_logits = self.linear_classifier_word(word_embedding_decoded)
        return word_logits
    
class Speaker_Decoder_Linear(torch.nn.Module):
    def __init__(self, num_speaker_class=200, d_model=704, hidden_size=512):
        super(Speaker_Decoder_Linear, self).__init__()
        self.num_speaker_class = num_speaker_class # change this after analyzing dataset
        # linear projection for speaker branch
        self.linear_project_speaker = nn.Linear(d_model,hidden_size)
        self.linear_classifier_speaker = nn.Linear(hidden_size,self.num_speaker_class)
        self.act = nn.ReLU()

    def forward(self, x):
        # Forward decoder pass on speaker branch
        # average across the frames of the speaker embedding 
        speaker_embedding = torch.mean(x,dim=1)
        speaker_embedding_decoded = self.act(self.linear_project_speaker(speaker_embedding))
        speaker_logits = self.linear_classifier_speaker(speaker_embedding_decoded)
        return speaker_logits


if __name__ == '__main__':
    model = Speech_Decoder_Linear()
    # x = {}
    x = torch.ones((2,1500,704))
    print(x)
    # compute the forward pass
    y1, y2 = model(x)
    # print(y.keys(), len(y["encoder_features_0"]), y["encoder_features_0"][1].shape)
    print(y1.shape)
    print(y2.shape)