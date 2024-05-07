import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

class Speech_Decoder_Linear(torch.nn.Module):
    def __init__(self,num_speaker_class=100,word_vocab=50,d_model=704):
        super(Speech_Decoder_Linear, self).__init__()
        self.num_speaker_class = num_speaker_class # change this after analyzing dataset
        self.word_vocab = word_vocab # change this after analyzing dataset -> len(vocab_dict)

        # Define downsampling layers
        # No downsampling needed as output of ECAPA-TDNN pooled and concatenating with the word embedding 
        
        # linear projection for speaker branch
        self.linear_project_speaker = nn.Linear(d_model,1)
        self.linear_classifier_speaker = nn.Linear(1,self.num_speaker_class) 

        # linear projection for word branch 
        self.linear_project_word = nn.Linear(d_model,1)
        self.linear_classifier_word = nn.Linear(1,self.word_vocab) 

    def forward(self, x):
        # Forward decoder pass on speaker branch
        speaker_embedding = x
        # average across the frames of the speaker embedding 
        speaker_embedding = torch.mean(speaker_embedding,dim=1)
        speaker_embedding = self.linear_project_speaker(speaker_embedding)
        speaker_logits = self.linear_classifier_speaker(speaker_embedding)
        
        # Forward decoder pass on word branch
        word_embedding = x
        # average across the frames of the speaker embedding 
        word_embedding = self.linear_project_word(word_embedding)
        word_logits = self.linear_classifier_word(word_embedding)

        return speaker_logits, word_logits


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