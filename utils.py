import re
import os
import yaml
import random
import logging

import torch
import torchaudio

import numpy as np
import pandas as pd

from pathlib import Path
import editdistance as ed
from easydict import EasyDict

import pytorch_lightning as pl
import torchaudio.transforms as T

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

def get_logger(name):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M', level=logging.DEBUG)
    logger = logging.getLogger(name)
    return logger

def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg
    
def extract_all_char(sentences):
    all_text = [phone for sentence in sentences for phone in sentence]
    vocab = list(set(all_text))
    return vocab

def prepare_data(batch, target_sr, tokenizer):
    #load audio files
    wav_file = batch["wav_path"]
    batch["audio"], orig_sr = torchaudio.load(wav_file)
    resampler = T.Resample(orig_sr, target_sr, dtype=batch["audio"].dtype)
    batch["audio"] = resampler(batch["audio"])

    tokens = tokenizer.encode(batch["sentence"])
    batch["text"] = tokens
    batch["text_length"] = len(tokens)
    
    return batch

def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(' ')
        t = t.split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot

if __name__ == '__main__':
    true = ['SH IY   HH AE D   Y UH R    D AA R K   S UW T   IH N G R IY S IY W AA SH W AO T ER AO L Y IH R', 'DH AH N UW S AH B ER B AH N AY T S W ER K T HH AA R D AA N R AH F ER B IH SH IH NG DH EH R OW L D ER HH OW M', 'Y UW K AE N B IH L D DH IH S V EY K EY SH AH N K AA T IH JH Y ER S EH L F', 'IH N F EH K SH AH S HH EH P AH T AY T AH S HH IY SH AW T IH D HH AA R T AH L IY', 'K AW N T DH AH N AH M B ER AH V T IY S P UW N Z AH V S OY S AO S DH AE T Y UW AE D', 'S M AE SH L AY T B AH L B Z AH N D DH EH R K AE SH V AE L Y UW W IH L D IH M IH N IH SH T UW N AH TH IH NG', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY S IY W AA SH W AO T ER AO L Y IH R', 'L AA T S AH V F AO R AH N M UW V IY Z HH AE V S AH B T AY T AH L Z', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY S IY W AA SH W AO T ER AO L Y IH R', 'T EY K CH AA R JH AH V CH UW Z IH NG HH ER B R AY D Z M EY D Z G AW N Z', 'D OW N T AE S K M IY T UW K AE R IY AE N OY L IY R AE G L AY K DH AE T', 'S AH M T AY M Z AO L DH OW B AY N OW M IY N Z AO L W EY Z DH IY Z AA R IH N D IY D AE L K AH L AY N', 'DH AH M IH S P R IH N T P R AH V OW K T AE N IH M IY D IY AH T D IH S K L EY M ER', 'W AH N M AO R M AH D AH L HH EH D IH D P L EY L AY K DH AE T W AH N AH N D DH EY D B IY L IY D AH NG HH IH M AH W EY', 'DH AH L AY B R EH R IY HH AE Z OW P AH N SH EH L V Z IY V IH N IH N DH AH AH N B AW N D P IH R IY AA D IH K AH L S T AA K R UW M', 'D OW N T AE S K M IY T UW K AE R IY AE N OY L IY R AE G L AY K DH AE T', 'AH L AW IY CH CH AY L D T UW HH AE V AE N AY S P OW P', 'K L IH R P R AH N AH N S IY EY SH AH N IH Z AH P R IY SH IY EY T IH D', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY S IY W AA SH W AO T ER AO L Y IH R', 'W AH N S Y UW F IH N IH SH G R IY S IH NG Y UH R CH EY N B IY SH UH R T UW W AA SH TH ER OW L IY', 'B R UW Z AH Z AH N D B L AE K AY Z W ER R IH L IY V D B AY AE P L AH K EY SH AH N AH V R AA B IY F S T EY K', 'DH IH S IH Z AH S L IY P IH NG K AE P S AH L', 'D AH Z HH IH N D UW AY D IY AA L AH JH IY AA N ER K AW Z', 'Y UW AO L W EY Z K AH M AH P W IH DH P AE TH AH L AA JH IH K AH L IH G Z AE M P AH L Z', 'S IY M S T R AH S IH Z AH T AE CH Z IH P ER Z W IH DH AH TH IH M B AH L N IY D AH L AH N D TH R EH D', 'SH EH L SH AA K K AA Z D B AY SH R AE P N AH L IH Z S AH M T AY M Z K Y UH R D TH R UW G R UW P DH EH R AH P IY', 'S OW R UW L Z W IY M EY D IH N AH N AH B AE SH T K AH L UW ZH AH N', 'B AH T T UW K AH N T IH N Y UW T UW D IH V AO R S AH D V AE N S T S T UW D AH N T S F R AH M R IY AE L AH T IY IH Z IH N AH K Y UW Z AH B AH L', 'HH AE V W IY N AA T AE K SH AH L IY D IH V EH L AH P T AY D IY AH W ER SH AH P', 'B R AH SH F AY ER Z AA R K AA M AH N IH N DH AH D R AY AH N D ER B R AH SH AH V N EH V AE D AH', 'AE Z K OW TH ER Z W IY P R IY Z EH N T AH D AA R N UW B UH K T UW DH AH HH AO T IY AA D IY AH N S', 'D OW N T AE S K M IY T UW K AE R IY AE N OY L IY R AE G L AY K DH AE T']
    pred = ['SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY S IY W AA SH W AO T ER AO L Y IH R', 'DH AH N UW S AH B ER B AH N AY T S W ER T HH AA R D AH N D R IY F V ER B IH AH SH IH NG DH EH R OW L D ER HH OW', 'Y UW K AE N B F IH L D DH IH V EY K EY SH AH N K AA T AH JH Y UH R S EH L F', 'IH N F EH K SH UH S P AH T AY T AH IH HH IY SH AW T IH D HH AA R T AH L IY', 'SH AW T DH AH N AH M B ER AH V T IY S P UW N Z AH V S OY IY S AO S D T AE T Y UW AE T', 'S M AE SH L AY B AA B P Z AH N DH EH K AE SH V AE AE L Y UW W IH L D IH M IH N IH SH T UW N AH TH IH NG', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY S IY W SH ER SH W AO T ER AO L Y IH R', 'DH AH L AA T S AH V F AO R AH N M UW V IY Z HH AE V S AH B T AY T AH L Z', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY IY W AO AA SH W AO T ER AO L Y IH R', 'T EY K CH AA R JH AH V CH UW Z IH NG HH ER B R R AY D Z M EY D Z G AW N', 'D OW N T AE S K M IY T UW K AE R IY AE N AA L IY R AE G L AY K DH AE T T', 'S AH M T AA AY M Z AO L DH OW B AY N OW M IY N Z AO L W EY Z DH IY Z ER R IH N D IY D L K AH L L N', 'DH AH M IH S P AE N T P R IH V OW K T AE N IH N M IY D IY AH T D IH S K L M ER', 'W AH N D M AO R M AH D AH L L HH D IH D P L EY L AY K DH AE T W AH N N DH EY D B IY L IY D IH NG N HH IH M M AH W EY', 'DH AH L AY B EH R IY HH AE Z OW P AH N SH EH L V Z IY V AH N IH N DH AH AH N B AW N P UH R IY AA T IH AH K AH L Z S T AA K R UW M', 'D OW N T AE S K M IY D T UW K AE IY AE N OW OY L IY R AE G L AY K DH AE T T', 'AH L AW IY CH CH AY L D T UW HH AE V AH N AA AY S P AA P', 'K L R ER P R AH N N S IY EY SH AH N IH Z AH P R IY SH IY EY T AH D', 'SH IY HH AE D Y UH R D AA R K S UW T IH N G R IY IY S IY W AA SH W AO T ER AO L Y IH IH R', 'AH N T L Y F IH N IH SH S G R IY S IH NG Y UH R CH EY N B IY SH UH R T UW W AA SH TH ER L IY', 'B R UW Z IH AH Z IH N D B L AE K AY Z W ER R IH L IY V B AY AE P L K EY SH N AH V ER R D B IY S S K', 'DH IH S IH Z AH S L IY P IH NG K AE P S S AH L', 'D AH Z HH IH N D UW AY D IY AA L AH JH IY AA N ER K AW Z', 'Y UW AO L W EY Z K AH M AH T P W IH DH P AE TH TH AH L AA JH IH K AH L G Z AE M P AH L Z', 'S IY M Z S ER IH Z AH T AE CH S IH P ER Z W IH DH AH TH EH M B AH L N IY D AH L AH N TH R EH D', 'SH EH L SH AA T K AA Z B AY S SH CH R AE P N L N S AH M T AY M Z K Y UH R D TH R G R UW P DH EH R AH P IY', 'S OW R UW L Z W ER M EY D IH N AH N AH B AE SH S D K AH L UW ZH SH AH N', 'B AH T T UW K AH N T IH N Y UW D T UW D IH V AO R S AE AH D V AE N S T UW AH N Z F R AH M R IY AE L AH D T IY IH Z IH N EH K S K Y UW Z AH B AH L L', 'HH AE V W IY N AA T AE K AH L IY D IH V EH L P P T AY D IY AH W ER SH P', 'ER AH SH F AY ER Z AA R K AA M AH N IH N DH AH D R AY AH N D ER B R SH SH AH V V AA D AH', 'AE Z K AH W AO R R TH Z W IY P R Z EH N T AH D AA R N UW B UH K K T UW DH AH HH AO T IY D IY AH N S', 'D OW N T AE S K M IY T UW K AE R IY AE N OY L IY R AE AE G L AY K DH AE T']
    print(cer(pred, true))
