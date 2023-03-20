import math
from pathlib import Path

import argbind
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import utils


class CollateFn:
    def __init__(self, device):
        self.device = device


    def __call__(self, samples):
        batch_size=len(samples)
        strokes, texts = zip(*samples)

        all_strokes = pad_sequence([torch.tensor(s, device=self.device) for s in strokes])
        all_texts = pad_sequence([torch.tensor(t, device=self.device) for t in texts])
        strokes_mask = torch.ones((all_strokes.shape[0], batch_size), dtype=torch.bool, device=self.device)
        texts_mask = torch.ones((all_texts.shape[0], batch_size), dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            texts_mask[len(texts[i]):,i] = 0
            strokes_mask[len(strokes[i]):,i] = 0

        return all_texts, texts_mask, all_strokes, strokes_mask


class HandwritingDataset(Dataset):
    @argbind.bind(without_prefix=True)
    def __init__(self, data_dir=Path('./descript-research-test/data'), is_validation=False, val_proportion=.1):
        super().__init__()

        self.strokes = np.load(data_dir / 'strokes-py3.npy', allow_pickle=True)
        sentences = (data_dir / 'sentences.txt').read_text().splitlines()
        assert len(sentences) == len(self.strokes)

        # 0 maps to empty character
        self.itoc = [''] + sorted(list({c for s in sentences for c in s}))
        self.ctoi = {c: i for i, c in enumerate(self.itoc)}

        self.sentences = [self.text2code(s) for s in sentences]

        num_validation = math.floor(len(self.strokes) * val_proportion)
        if num_validation > 0:
            if is_validation:
                self.sentences = self.sentences[-num_validation:]
                self.strokes = self.strokes[-num_validation:]
            else:
                self.sentences = self.sentences[:-num_validation]
                self.strokes = self.strokes[:-num_validation]

    def text2code(self, s):
        return np.array([self.ctoi[c] for c in s], dtype=np.int32)

    def numCharacters(self):
        return len(self.ctoi)

    def code2text(self, s):
        return ''.join([self.itoc[i] for i in np.nditer(s)])

    def getSentences(self):
       return [self.text2code(s) for s in self.sentences]

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        return self.strokes[idx], self.sentences[idx]


@argbind.bind(without_prefix=True)
def main(show_regular=True, show_inf_dl=False, data_device: str = "cpu"):
    if show_regular:
        dataset = HandwritingDataset()
        collate_fn = CollateFn(data_device)
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

    if show_inf_dl:
        dataset = HandwritingDataset(is_validation=True, validation_percentage=.02)
        collate_fn = CollateFn(data_device)
        dl = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)
        print(f"num batches in dl: {len(list(dl))}")
        dl = utils.infinite_dl(dl)
        max_step = 100
        step = 0

        for texts, texts_mask, strokes, strokes_mask in dl:
            step += 1
            if step > max_step:
                break
            print(f"text from batch {step}: {dataset.code2text(texts[:,0].numpy())}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
