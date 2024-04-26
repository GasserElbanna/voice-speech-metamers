from collections import defaultdict

import torch
from transformers import Wav2Vec2Model
from speechbrain.inference.encoders import WaveformEncoder


class Encoder(torch.nn.Module):
    def __init__(self, model_name, ckpt, cache_dir):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "wav2vec2_orig":
            self.model = Wav2Vec2Model.from_pretrained(ckpt, cache_dir=cache_dir)
        else:
            self.model = WaveformEncoder.from_hparams(source=ckpt, savedir=cache_dir, run_opts={"device":"cuda"})
            self.activation = defaultdict(list)
            for layer in range(12):
                getattr(self.model.mods.encoder.encoder_wrapper.latent_encoder.layers, str(layer)).register_forward_hook(self.get_layer_embeddings(f'encoder_features_{layer}'))

    def get_layer_embeddings(self, name):
        def hook(model, input, output):
            self.activation[name].append(output[0])
        return hook
    
    def forward(self, input_values):
        device = input_values[0].device
        if self.model_name == "wav2vec2_orig":
            output_values = [self.model(input_.squeeze(0), output_hidden_states=True) for input_ in input_values]
            return [list(output_.hidden_states[1:]) for output_ in output_values]
        else:
            inputs_len = torch.FloatTensor([len(input_) for input_ in input_values]).to(device)
            inputs_len_norm = inputs_len/inputs_len.max()
            _ = [self.model.encode_batch(input_.squeeze(0), inputs_len_norm[i])["embeddings"].squeeze(0) for i, input_ in enumerate(input_values)]
            # return self.activation
            output_values = []
            for i in range(len(input_values)):
                output_values.append([self.activation[name][i] for name in self.activation.keys()])
            self.activation = defaultdict(list)
            return output_values
            

if __name__ == '__main__':
    model = Encoder("wav2vec2_sb", "/om2/user/gelbanna/raw_wav2vec2/save/CKPT+2022-09-20+03-01-00+00", "/om2/user/gelbanna/raw_wav2vec2")
    # model = Encoder("wav2vec2_orig", "facebook/wav2vec2-base", "/om2/user/gelbanna/model_cache")
    # x = {}
    x = [torch.ones((1,1,35000)), torch.ones((1,1,30000))]
    print(x)
    # compute the forward pass
    y = model(x)
    # print(y.keys(), len(y["encoder_features_0"]), y["encoder_features_0"][1].shape)
    print(len(y))
    print(len(y[0]), len(y[1]))
    print(y[0][0].shape, y[1][0].shape)