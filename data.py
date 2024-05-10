import torch
from dataclasses import dataclass

@dataclass
class DataCollator:
    """
    a variant of callate_fn that pads inputs in a batch
    """

    def __call__(self, batch):
        batch_dict = {}
        batch_dict['input_values'] = [torch.FloatTensor(feature["audio"]).unsqueeze(0) for feature in batch]
        batch_dict['speaker_labels'] = torch.LongTensor([feature["speaker_int"] for feature in batch])
        batch_dict['speech_labels'] = [torch.LongTensor(feature["text"]) for feature in batch]
        batch_dict['sentence'] = [feature["sentence"] for feature in batch]

        return batch_dict