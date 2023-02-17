import argbind
from datetime import datetime
from pathlib import Path
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter

import data
from model import Scribe
import utils

@argbind.bind(without_prefix=True)
def train(
        save_path: Path = Path('runs') / datetime.now().isoformat(),
        num_training_iterations: int = 1000,
        batch_size: int = 40,
        device: str = "cpu",
        random_seed: int = None):

    """Trains model, samples an output, and displays it

    Parameters
    ----------
    save_path : Path
        The location of the tensorboard run (defaults to runs/CURDATETIME)
    device: str
        Which GPU device to run on ("cuda:0", "cuda:1", etc.) or "cpu" to run on CPU
    """
    writer = SummaryWriter(save_path)
    if random_seed:
        print(f"Setting random seed: {random_seed}")
        torch.manual_seed(random_seed)
    dataset = data.HandwritingDataset()
    dl = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=data.collate_fn)
    dl = utils.infinite_dl(dl)

    model = Scribe()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())

    iteration_times = []
    for iteration, batch in enumerate(dl):
        texts, texts_mask, strokes, strokes_mask = batch
        if iteration >= num_training_iterations:
            break
        optimizer.zero_grad()
        start = time.time()
        strokes = strokes.to(device)
        output = model(strokes)
        iteration_times.append(time.time() - start)

        # output[0] is the prediction, strokes[1] is the ground truth
        loss = ((strokes[1:] - output[:-1]) ** 2).mean()
        if (iteration % 10) == 0:
            print(f"iter {iteration:4d}: {loss.item()=:.4f}")
        writer.add_scalar('Loss/train', loss.item(), iteration)
        loss.backward()
        optimizer.step()
    print(f"mean iteration time: {torch.tensor(iteration_times).mean():.4f}")

    # evaluation
    with torch.no_grad():
        model.eval()
        sample = model.sample(batch_size=1)
        utils.plot_stroke(sample[:, 0].to("cpu"))


def main():
    args = argbind.parse_args()
    args["save_path"].mkdir(exist_ok=True, parents=True)
    argbind.dump_args(args, args["save_path"] / "args.yml")

    with argbind.scope(args):
        train()


if __name__ == "__main__":
    main()
