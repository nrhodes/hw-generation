import time
from datetime import datetime
from pathlib import Path

import argbind
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data
import utils
from model import Scribe


class ScribeLoss:
    @argbind.bind
    def __init__(self, penup_weighting=.5):
        self.penup_weighting = penup_weighting
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, prediction, target, target_mask, penup_weighting=.5):
        penup_loss = self.bce_loss(prediction[:, :, 0:1], target[:, :, 0:1]).squeeze()
        coords_loss = self.mse_loss(prediction[:, :, 1:3], target[:, :, 1:3])
        coords_loss = coords_loss.sum(dim=2)
        loss =  penup_weighting*penup_loss + (1-penup_weighting)*coords_loss
        loss = loss.masked_select(target_mask)
        return loss.mean()


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
    collate_fn = data.CollateFn(device)
    dl = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    dl = utils.infinite_dl(dl)

    model = Scribe()
    model = model.to(device)
    scribe_loss = ScribeLoss()

    optimizer = optim.Adam(model.parameters())

    iteration_times = []
    with tqdm(total=num_training_iterations) as pbar:
        for iteration, batch in enumerate(dl):
            texts, texts_mask, strokes, strokes_mask = batch
            if iteration >= num_training_iterations:
                break
            optimizer.zero_grad()
            start = time.time()
            strokes = strokes
            output = model(strokes)
            iteration_times.append(time.time() - start)

            # output[:-1] is the prediction, strokes[1:] is the ground truth
            loss = scribe_loss(output[:-1], strokes[1:], strokes_mask[1:])
            writer.add_scalar('Loss/train', loss.item(), iteration)
            loss.backward()
            optimizer.step()
            if (iteration % 100) == 0:
                with torch.no_grad():
                    model.eval()
                    sample = model.sample()
                    f = utils.plot_stroke(sample[:, 0].to("cpu"), "xyz.png")
                    writer.add_figure(f"sample, bias: 0", figure=f, global_step=iteration)
                    model.train()

            pbar.postfix = f": Loss: {loss.item():.4f}"
            pbar.update()

    torch.save(model.state_dict(), save_path / "model.pt")
    print(f"mean iteration time: {torch.tensor(iteration_times).mean():.4f}")

    # evaluation
    with torch.no_grad():
        model.eval()
        for bias in [0, .1, .5, 2, 5, 10]:
            sample = model.sample(bias=bias)
            f = utils.plot_stroke(sample[:, 0].to("cpu"), "xyz.png")
            writer.add_figure(f"sample, bias: {bias}", f, global_step=num_training_iterations)
        # for some reason, last added figure never shows up
        writer.add_figure(f"sacrificial figure", f)



def main():
    args = argbind.parse_args()
    args["save_path"].mkdir(exist_ok=True, parents=True)
    argbind.dump_args(args, args["save_path"] / "args.yml")

    with argbind.scope(args):
        train()


if __name__ == "__main__":
    main()
