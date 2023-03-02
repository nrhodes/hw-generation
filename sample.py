from pathlib import Path

import argbind
import torch

import utils
from model import Scribe


@argbind.bind(without_prefix=True)
def sample(
        model_weights: Path = None,
        device: str = "cpu",
        random_seed: int = None):

    """Samples

    Parameters
    ----------
    model_weights: Path
        The location of the saved weights
    device: str
        Which GPU device to run on ("cuda:0", "cuda:1", etc.) or "cpu" to run on CPU
    random_seed: int
        Random seed
    """
    if random_seed:
        print(f"Setting random seed: {random_seed}")
        torch.manual_seed(random_seed)

    model = Scribe()
    model.load_state_dict(torch.load(model_weights))
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        sample = model.sample()
        f = utils.plot_stroke(sample[:, 0].to("cpu"))


def main():
    args = argbind.parse_args()
    with argbind.scope(args):
        sample()


if __name__ == "__main__":
    main()
