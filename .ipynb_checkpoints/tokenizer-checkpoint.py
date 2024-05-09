class Tokenizer:
    def __init__(self, vocab_file, pad='<pad>', eos='<eos>', unk='<unk>'):

        vocab_list = self.load_from_file(vocab_file)
        vocab_list[-1] = ' '
        self._vocab_list = ["<pad>", "<eos>", "<unk>"] + vocab_list
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}
    
    @property
    def pad_idx(self):
        return 0

    @property
    def eos_idx(self):
        return 1

    @property
    def unk_idx(self):
        return 2
    
    @property
    def vocab_size(self):
        return len(self._vocab_list)
        
    def load_from_file(self, vocab_file):
        with open(vocab_file, "r") as f:
            # Do not strip space because character based text encoder should
            # have a space token
            vocab_list = [line.strip("\r\n") for line in f]
        return vocab_list
    
    def vocab_to_idx(self, vocab):
        return self._vocab2idx.get(vocab, self.unk_idx)
    
    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]
    
    def encode(self, phones):
        # Manually append eos to the end
        return [self.vocab_to_idx(phone) for phone in phones] + [self.eos_idx]

    def decode(self, idxs, ignore_repeat=False):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.eos_idx:
                break
            elif idx == self.pad_idx or (ignore_repeat and t > 0 and idx == idxs[t-1]):
                continue
            else:
                vocabs.append(v)
        return " ".join(vocabs)
        # return vocabs

if __name__ == '__main__':
    tokenizer = Tokenizer('/om2/user/gelbanna/finetuning_phoneme_recognition/phonemes.txt')

    text1 = ['SH', 'IY', ' ', 'HH', 'AE', 'D', ' ', 'Y', 'UH', 'R', ' ', 'D', 'AA', 'R', 'K', ' ', 'S', 'UW', 'T', ' ', 'IH', 'N', ' ', 'G', 'R', 'IY', 'S', 'IY', ' ', 'W', 'AA', 'SH', ' ', 'W', 'AO', 'T', 'ER', ' ', 'AO', 'L', ' ', 'Y', 'IH', 'R']
    text2 = ['SH', 'IY', ' ', 'HH', 'AE', 'D', ' ', 'Y', 'UH', 'R', ' ', 'D']

    print(tokenizer._vocab2idx)
    print(tokenizer.encode(text1))
    x = tokenizer.decode(tokenizer.encode(text1))
    y = tokenizer.decode(tokenizer.encode(text2))
    print(x.split())
    from utils import per

    print(per(x.split(),y.split()))
    