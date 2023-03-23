import math

import argbind
import torch
import torch.nn.functional as F
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
        masked_embedding = embedded_text * text_mask.unsqueeze(-1)
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
        weightedSum = masked_embedding * weights
        new_context = weightedSum.sum(dim=1)

        return new_context, mean

class MixtureLayer(nn.Module):
    def __init__(self, num_inputs, num_mixtures=3, epsilon=1e-6):
        super(MixtureLayer, self).__init__()
        self.num_mixtures = num_mixtures
        self.linear = nn.Linear(in_features=num_inputs, out_features=1+5*num_mixtures)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.epsilon = epsilon

    def forward(self, x):
        # [:, :, 0:1] param is penup logit
        # [:, :, 1:1+num_mixtures] are mixture weights
        # [:, :, 1+1*num_mixtures:1 + 3*num_mixtures] are x/y coords
        # [:, :, 1+3*num_mixtures:1 + 4*num_mixtures] are stddevs
        #
        return self.linear(x)

    def get_distribution_parameters(self, x):
        penup, log_weights, means, log_std =  x.split([1, self.num_mixtures, 2*self.num_mixtures, 2*self.num_mixtures], dim=-1)

        means = means.view(means.shape[:-1] + (self.num_mixtures, 2))
        log_std = log_std.view(log_std.shape[:-1] + (self.num_mixtures, 2))
        log_weights = F.log_softmax(log_weights, dim=-1)

        # Hardcode rho to 0 until I figure out how to sample correctly
        rho = torch.zeros(means.shape[:-1], device=means.device)
        penup = torch.sigmoid(penup)
        return penup, log_weights, means, log_std, rho

    def get_sample(self, x, bias=10):
        penup_prob, log_weights, means, log_std, rho = self.get_distribution_parameters(x)

        weights = torch.softmax(torch.exp(log_weights*(1+bias)), dim=-1).squeeze(dim=0)
        which_mixture = torch.multinomial(weights, 1).squeeze(0)
        log_std = log_std[:, :, which_mixture, :]
        means = means[:, :, which_mixture, :]
        rho = rho[:, :, which_mixture]

        stds = torch.exp(log_std - bias)

        # sample
        mu1, mu2 = means.unbind(-1)
        std1, std2 = stds.unbind(-1)

        rand1 = torch.randn_like(mu1)
        rand2 = torch.randn_like(mu2)
        coord1 = rand1 * std1 + mu1
        coord2 = rand2 * std2 + mu2
        penup = penup_prob.bernoulli()
        sample = torch.cat((penup, coord1, coord2), dim=-1)
        return sample

    def get_scribe_loss(self, prediction, target, target_mask=None):
        penup, log_weights, means, log_std, rho = self.get_distribution_parameters(prediction)
        target_penup, x, y = target.unbind(-1)
        #print(f"{target_penup=}")
        penup_loss = F.binary_cross_entropy(penup.squeeze(-1), target_penup)
        penup_loss = penup_loss.masked_select(target_mask).mean()

        mu1, mu2 = means.unbind(-1)
        logstd1, logstd2 = log_std.unbind(-1)
        std1 = torch.exp(logstd1) + self.epsilon
        std2 = torch.exp(logstd2) + self.epsilon

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        frac1 = (x - mu1)/std1
        frac2 = (y - mu2)/std2
        Z = frac1**2 + frac2**2- 2*rho*frac1*frac2
        logN = -Z/2*(1-rho**2 + self.epsilon) - torch.log(torch.tensor([2*math.pi], device=x.device)) - logstd1 - logstd2 -0.5 * torch.log(1-rho**2 + self.epsilon)

        log_coords_loss = -torch.logsumexp(logN + log_weights, dim=-1)
        log_coords_loss = log_coords_loss.masked_select(target_mask).mean()

        loss =  penup_loss + log_coords_loss
        return loss


class Scribe(nn.Module):
    @argbind.bind(without_prefix=True)
    # TODO(neil): Update num_embeddings from data
    def __init__(self, dataset, input_size=3, hidden_size=20):
        super(Scribe, self).__init__()
        self.embedding_dim = 10
        self.dataset = dataset
        self.embedding = nn.Embedding(num_embeddings=dataset.numCharacters(), embedding_dim=self.embedding_dim)
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.rnn2 = nn.GRU(input_size=hidden_size+self.embedding_dim, hidden_size=hidden_size)
        self.attention = GaussianAttention(query_size=hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.mixture_layer = MixtureLayer(hidden_size)

    def get_scribe_loss(self, prediction, target, target_mask=None):
        return self.mixture_layer.get_scribe_loss(prediction, target, target_mask)

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
        output = self.mixture_layer(o2)

        result = output, (hidden1, previous_position, hidden2)
        return result

    @property
    def device(self):
        return next(self.parameters()).device

    @argbind.bind
    def sample(self, promptStr, num_steps=120, bias=0.0, stddev=0.1):
        batch_size = 1
        embedded_prompt = self.embedding(torch.tensor(self.dataset.text2code(promptStr), device=self.device).unsqueeze(dim=1))
        prompt_mask = torch.ones((len(promptStr), 1), device=self.device)
        sample = torch.zeros(1, batch_size, 3, device=self.device)
        hidden = None
        stddev = torch.log(torch.tensor([stddev, stddev]).view(1, batch_size, 2).to(self.device))
        output = []
        for _ in range(num_steps):
            x, hidden = self.step(sample, embedded_prompt, prompt_mask, hidden)

            sample = self.mixture_layer.get_sample(x, bias)
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
