from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch import nn
import torch

import data
import utils

class Scribe(nn.Module):
    def __init__(self, input_size=3, hidden_size=20, output_size=3):
        super(Scribe, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, strokes):
        hidden = None
        output, hidden = self.step(strokes, hidden)
        return output

    def step(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        output = self.linear(x)
        return output, hidden

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, batch_size=1, num_steps=100):
        x = torch.zeros(1, batch_size, 3).to(self.device)
        hidden = None
        output = []
        for i in range(num_steps):
            x, hidden = self.step(x, hidden)
            output.append(x)
        return torch.vstack(output)




def main():
    parser = ArgumentParser(prog="model")
    parser.add_argument("--show_strokes", action='store_true')
    parser.add_argument("--show_output", action='store_true')
    parser.add_argument("--show_samples", action='store_true')
    args = parser.parse_args()

    dataset = data.HandwritingDataset()
    dl = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=data.collate_fn)
    texts, texts_mask, strokes, strokes_mask = next(iter(dl))

    model = Scribe()

    with torch.no_grad():
        model.eval()

        output = model(strokes)
        print(f"{output.shape=}")

        if args.show_strokes:
            print("strokes")
            for i in range(strokes.shape[1]):
                utils.plot_stroke(strokes[:,i])

        if args.show_output:
            print("outputs")
            for i in range(output.shape[1]):
                utils.plot_stroke(output[:,i])

        if args.show_samples:
            sample = model.sample(batch_size=3) # since our sampling has no stochasticity, sample will always return the same sample
            print("samples (shape: {sample.shape})")
            for i in range(sample.shape[1]):
                utils.plot_stroke(sample[:,i])


if __name__ == "__main__":
    main()
