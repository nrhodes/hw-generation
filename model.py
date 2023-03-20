from argparse import ArgumentParser

import argbind
import torch
from torch import nn
from torch.utils.data import DataLoader

import data
import utils


class GaussianAttention(nn.Module):
    def __init__(self, query_size=5):
        super(GaussianAttention, self).__init__()
        self.linear = nn.Linear(in_features=query_size, out_features=2)

    def forward(self, query, embedded_text, text_mask, position):
        """query should be of shape [seq_len, batch_size, query_size]]
        textEmbedding should be of shape [text_len, batch_size]
        text_mask should be of shape [text_len, batch_size]
        position should be of shape [seq_len, batch_size] (or None)
        Returns tuple of:
          * a tensor of shape [seq_len, batch_size, embedding_dim]
          * a tensor of shape [seq_len, batch_size] containing positions."""
        # TODO(neil): rewrite so that we don't need seq_len leading dimension
        maskedEmbedding = embedded_text * text_mask.unsqueeze(-1)
        seq_len = query.shape[0]
        batch_size = query.shape[1]
        if position is None:
            position = torch.zeros((seq_len, batch_size), device=query.device)
        query_result = torch.exp(self.linear(query))
        assert query_result.shape == (seq_len, batch_size, 2)
        assert embedded_text.shape[1] == batch_size
        mean, std = query_result.unbind(dim=-1)

        # We divide by 20 because avg number of strokes/character is 20.
        mean = position + mean.cumsum(dim=0)/20
        # TODO(neil): Why these extra dimensions with value?
        indices = torch.arange(0, embedded_text.shape[0], device=query.device).view(1, embedded_text.shape[0], 1, 1)
        weights = torch.exp(-std.view(seq_len, 1, batch_size, 1) * (mean.view(seq_len, 1, batch_size, 1) - indices) ** 2)
        weightedSum = maskedEmbedding * weights
        newContext = weightedSum.sum(dim=1)

        return newContext, mean

class Scribe(nn.Module):
    @argbind.bind(without_prefix=True)
    # TODO(neil): Update num_embeddings from data
    def __init__(self, dataset, input_size=3, hidden_size=20, output_size=3):
        super(Scribe, self).__init__()
        self.embedding_dim = 10
        self.dataset = dataset
        self.embedding = nn.Embedding(num_embeddings=dataset.numCharacters(), embedding_dim=self.embedding_dim)
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.rnn2 = nn.GRU(input_size=hidden_size+self.embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.attention = GaussianAttention(query_size=hidden_size)


    def forward(self, strokes, texts, textsmask):
        hidden = None
        embedded_text = self.embedding(texts)
        output, hidden = self.step(strokes, embedded_text, textsmask, hidden)
        return output

    def step(self, x, embedded_text, textsmask, hidden):
        if hidden is None:
            hidden1 = None
            previous_position = None
            hidden2 = None
        else:
            hidden1, previous_position, hidden2 = hidden
        o1, hidden1 = self.rnn1(x, hidden1)
        context, previous_position = self.attention(o1, embedded_text, textsmask, previous_position)
        o2, hidden2 = self.rnn2(torch.cat((o1, context), dim=-1), hidden2)
        output = self.linear(o2)

        result = output, (hidden1, previous_position, hidden2)
        return result

    @property
    def device(self):
        return next(self.parameters()).device

    @argbind.bind
    def sample(self, promptStr, num_steps=120, bias=0.0, stddev=0.1):
        def sample_from_distribution(mean, std_logits):
            std = torch.exp(std_logits - bias)
            return torch.normal(mean=mean, std=std)

        batch_size = 1
        promptEmbedding = self.embedding(torch.tensor(self.dataset.text2code(promptStr), device=self.device).unsqueeze(dim=1))
        promptMask = torch.ones((len(promptStr), 1), device=self.device)
        sample = torch.zeros(1, batch_size, 3, device=self.device)
        hidden = None
        stddev = torch.log(torch.tensor([stddev, stddev]).view(1, batch_size, 2).to(self.device))
        output = []
        for _ in range(num_steps):
            x, hidden = self.step(sample, promptEmbedding, promptMask, hidden)

            mean = x[:, :, 1:3]
            coords = sample_from_distribution(mean, stddev)
            penup = torch.sigmoid(x[:, :, 0:1]) > torch.rand((1, batch_size, 1)).to(self.device)
            sample = torch.cat((penup, coords), dim=2)
            output.append(sample)
        return torch.vstack(output)



@argbind.bind(without_prefix=True)
def main(show_strokes=True, show_output=False, show_samples=False):
    dataset = data.HandwritingDataset()
    dl = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=data.collate_fn)
    texts, texts_mask, strokes, strokes_mask = next(iter(dl))

    # TODO: Apply mask to strokes
    model = Scribe(dataset)

    with torch.no_grad():
        model.eval()

        output = model(strokes)
        print(f"{output.shape=}")

        if show_strokes:
            print("strokes")
            for i in range(strokes.shape[1]):
                utils.plot_stroke(strokes[:,i])

        if show_output:
            print("outputs")
            for i in range(output.shape[1]):
                utils.plot_stroke(output[:,i])

        if show_samples:
            sample = model.sample(batch_size=3)
            print("samples (shape: {sample.shape})")
            for i in range(sample.shape[1]):
                utils.plot_stroke(sample[:,i])


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
