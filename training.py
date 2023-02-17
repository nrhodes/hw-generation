from argparse import ArgumentParser
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter


import data
from model import Scribe
import utils

def train(model, dl, num_training_iterations):
    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters())
    for iteration, batch in enumerate(dl):
        texts, texts_mask, strokes, strokes_mask = batch
        if iteration >= num_training_iterations:
            break
        optimizer.zero_grad()
        output = model(strokes)

        # output[0] is the prediction, strokes[1] is the ground truth
        loss = ((strokes[1:] - output[:-1]) ** 2).mean()
        if (iteration % 10) == 0:
            print(f"{loss.item()=:.4f}")
        writer.add_scalar('Loss/train', loss.item(), iteration)
        loss.backward()
        optimizer.step()


def main():
    parser = ArgumentParser(prog="train")
    #parser.add_argument("--test_unrolled", action=argparse.BooleanOptionalAction)
    parser.add_argument("--random_seed", default=None, type=int)
    parser.add_argument("--num_training_iterations", default=1000, type=int)
    args = parser.parse_args()

    if args.random_seed:
        torch.manual_seed(args.random_seed)
    dataset = data.HandwritingDataset()
    dl = DataLoader(dataset, shuffle=True, batch_size=40, collate_fn=data.collate_fn)
    inf = data.infinite_dl(dl)

    model = Scribe()
    train(model, inf, args.num_training_iterations)
    with torch.no_grad():
        model.eval()
        sample = model.sample(batch_size=1)
        utils.plot_stroke(sample[:, 0])

if __name__ == "__main__":
    main()
