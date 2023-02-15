from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import math
import numpy as np
import torch
import utils

def collate_fn(samples):
    batch_size=len(samples)
    strokes = [s for s, text in samples]
    texts = [text for s, text in samples]

    max_stroke_len = max([len(s) for s in strokes])
    num_stroke_elements=strokes[0].shape[1]
    all_strokes = torch.zeros((max_stroke_len, batch_size, num_stroke_elements))
    strokes_mask = torch.ones((max_stroke_len, batch_size), dtype=torch.bool)
    for i in range(batch_size):
        this_stroke_len = strokes[i].shape[0]
        # pad
        all_strokes[:this_stroke_len,i] = torch.tensor(strokes[i])
        # mask
        strokes_mask[this_stroke_len:,i] = 0

    max_texts_length = max([len(t) for t in texts])
    all_texts = torch.zeros((max_texts_length, batch_size), dtype=torch.int8)
    texts_mask = torch.ones((max_texts_length, batch_size), dtype=torch.bool)
    for i in range(batch_size):
        this_texts_len = texts[i].shape[0]
        # pad
        all_texts[:this_texts_len,i] = torch.tensor(texts[i])
        # mask
        texts_mask[this_texts_len:,i] = 0

    return all_texts, texts_mask, all_strokes, strokes_mask
    
class HandwritingDataset(Dataset):
    def __init__(self, is_validation=False, validation_percentage=.1, data_dir=Path('./descript-research-test/data')):
        super().__init__()
        self.strokes = np.load(data_dir / 'strokes-py3.npy', allow_pickle=True)
        sents = (data_dir / 'sentences.txt').read_text().splitlines()
        self.sentences = [self.text2code(s) for s in sents]
        assert len(self.sentences) == len(self.strokes)

        num_validation = math.floor(len(self.strokes) * validation_percentage)
        if is_validation:
            self.sentences = self.sentences[-num_validation:]
            self.strokes = self.strokes[-num_validation:]
        else:
            self.sentences = self.sentences[:-num_validation]
            self.strokes = self.strokes[:-num_validation]

    def char2index(self, ch):
        return ord(ch)

    def index2char(self, i):
        return chr(i)

    def text2code(self, s):
        return np.array([self.char2index(c) for c in s], dtype=np.int16)

    def code2text(self, s):
        return ''.join([self.index2char(i) for i in np.nditer(s)])

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        return self.strokes[idx], self.sentences[idx]


def main():
    dataset = HandwritingDataset()

    dl = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)
    texts, texts_mask, strokes, strokes_mask = next(iter(dl))

    print(f"texts.shape: {texts.shape}")
    print(f"texts_mask.shape: {texts_mask.shape}")
    print(f"strokes.shape: {strokes.shape}")
    print(f"strokes_mask.shape: {strokes_mask.shape}")

    for i in range(texts.shape[1]):
        print(f"text {i}: {dataset.code2text(texts[:,i].numpy())}")
        print(f"text mask {i}: {texts_mask[:,i]}")
        utils.plot_stroke(strokes[:,i])


if __name__ == "__main__":
    main()
