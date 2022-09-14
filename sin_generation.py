# -*- coding: utf-8 -*-
# We want to predict sin waves
# Given 5 previous values, predict 6th?
# Reuse hidden state information?
# Do 
3+4
# %%
import io
import matplotlib.pyplot as pyplot
from pathlib import Path
from pytorch_lightning.callbacks import Callback

from torchvision import transforms

# %%
import requests
hwdir = Path('/data') / 'neil' / 'hw'

args = {
    # for model
    'epochs':40,
    'batch_size': 20,
    'lr': .003,
    "rnn_hidden_size": 200,
    "training_noise_variance": 0.00,
    "generation_noise_variance": 0.01,
    "seq_len": 20,
    "samples_per_cycle": 20,
    "cycles": 5,

    # for generation
    "generated_length": 80,
}

# %%
"""# Initialization"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl

#torch.manual_seed(args["seed"])
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")




import numpy as np

# %%
# Assume prompt is from 0, 
def plot_stroke(prompt, y):
    f, ax = pyplot.subplots()

    prompt_samples = prompt.shape[0]
    y_samples = y.shape[0]
    tot_samples = prompt_samples + y_samples
    upper_bound = (tot_samples / args["samples_per_cycle"]) * 2 * math.pi
    x = np.linspace(0.0, upper_bound, num=tot_samples)

    sin = np.sin(x)
    #print('x', x)
    #print('y', y)
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    #f.set_size_inches(5. * size_x / size_y, 5.)
    f.set_size_inches(7.0, 2.0)

    ax.plot(x, sin, 'r-', linewidth=1)
    ax.plot(x[:prompt_samples], prompt, 'b-', linewidth=1)
    ax.plot(x[prompt_samples:], y, 'k-', linewidth=2)

    #ax.axis('equal')
    ax.set(ylim=(-1.0, 1.0))
    ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)

    pyplot.show()

    return f

x = model.train_ds[0][0]
y=model.train_ds[0][1]
prompt= x
prediction = model.generate_unconditionally(prompt=prompt, noise_variance=.01, verbose=True)

print(x.shape, y.shape, prediction.shape)
print(f'x: {x}')
print(f'y: {y[-5:]}')
print(f'prediction: {prediction[:3]}')
plot_stroke(prompt, prediction)


# %%

from torch.utils.data import DataLoader
import numpy as np
import math

class SinDataset(Dataset):
    def __init__(self, seq_len=args["seq_len"], cycles=args["cycles"], samples_per_cycle=args["samples_per_cycle"]):
        self.num_samples=cycles*samples_per_cycle
        self.seq_len=seq_len
        xs = np.linspace(0.0, math.pi*2*cycles, num=self.num_samples, dtype=np.float32)
        self.sins = np.sin(xs)
        self.sins = torch.as_tensor(self.sins)
        self.sins = torch.reshape(self.sins, (self.num_samples, 1))
        #print(f"self.sins: {self.sins}")

    def __len__(self):
        return self.num_samples-1-self.seq_len

    def __getitem__(self, idx):
        return self.sins[idx:self.seq_len+idx,:], self.sins[idx+1:self.seq_len+idx+1, :]

# %%

d = SinDataset()
import inspect
print(len(d))
d[0]
inspect.getmembers(d)

# %%
import string

class HWModel(pl.LightningModule):
  def __init__(self, lr=args['lr'], bs = args['batch_size']):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.rnn = nn.GRU(input_size=1, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 1) 

    self.learning_rate = lr
    self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
    self.bs = bs

    # Return the hidden tensor(s) to pass to forward
  def get_new_hidden(self, batch_size):
    return torch.zeros(1,  batch_size, self.hidden_size, device=self.device)

  def forward(self, x, hidden):
    VERBOSE=False
    if VERBOSE:
        print(f'Forward: input: {x.shape}, hidden: {hidden.shape}')

    (x, hidden) = self.rnn(x, hidden)

    if VERBOSE:
      print(f'Forward: size after rnn: {x.size()}')
      #print(f'Forward: after rnn: {x} ')
    x = self.fc(x)
    if VERBOSE:
      print(f'Forward: size after fc: {x.size()}')
      #print(f'Forward:  after fc: {x}')
    return x, hidden
    
  def loss(self, y_hat, y):
      #print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
      #print(f'y_hat, {y_hat}')
      #print(f'y    , {y}')
      mse_loss = F.mse_loss(y_hat, y) 
      #print(f'mse_loss: {mse_loss}, bce_loss = {bce_loss}')
      return {
              'total': mse_loss,
      }

  def training_step(self, batch, batch_idx, hiddens=None):
    data, y = batch
    if args['training_noise_variance'] > 0:
        data = data + torch.randn(data.shape, device=self.device) * args['training_noise_variance']
        y = y + torch.randn(y.shape, device=self.device) * args['training_noise_variance']
    if hiddens is None:
        hiddens = self.get_new_hidden(batch_size=data.shape[0])
    y_hat, hiddens = self(data, hiddens)

    losses = self.loss(y_hat, y)
    if batch_idx == 0 and not self.trainer.auto_lr_find:
        print(f'epoch: {self.current_epoch} batch: {batch_idx} training loss({losses})')
        self.log('loss_train', losses, prog_bar=True)
        x = model.train_ds[0][0]
        sample = self.generate_unconditionally(prompt=x)
        #print(f'sample: {sample}')
        f = plot_stroke(x, sample)
        self.logger.experiment.add_figure('generated sin', f, self.current_epoch)

    if not self.trainer.auto_lr_find:
        self.logger.experiment.add_scalars("losses", {"train_loss": losses["total"]})
    return {'loss': losses['total']}

  def generate_unconditionally(self, prompt=None, output_length=args["generated_length"], 
            noise_variance=args["generation_noise_variance"], verbose=False):
    hidden = self.get_new_hidden(batch_size=1)
    output = torch.zeros((output_length+1), device=self.device)
    noise = torch.randn((output_length+1), device=self.device) * noise_variance
    prompt = prompt.type_as(hidden)

    if prompt is not None:
      with torch.no_grad():
        input = torch.unsqueeze(prompt, 0)
        if verbose:
            print(f"prompt: input to forward: {input}")
        predictions, hidden = self.forward(input, hidden)
        if verbose:
            print(f"predictions: {predictions}")
        output[0] = predictions[0, -1]

    for idx in range(output_length):
      with torch.no_grad():
        input = output[idx]
        if verbose:
            print(f"input to forward: {input}")
        predictions, hidden = self.forward(torch.reshape(input, (1, 1, 1)), hidden)
        if verbose:
            print(f"output from forward: {predictions}")
      output[idx+1] = predictions[0, -1]

      # Do sampling from the prediction:
      output[idx+1] = output[idx+1] + noise[idx]

    if prompt is  None:
        #skip first (zero) element
        output = output[1:]
    asNumpy = output.cpu().numpy()
    return asNumpy

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return [optimizer], []

  def prepare_data(self):
      pass 


  def train_dataloader(self):
      return DataLoader(self.train_ds, shuffle=False, batch_size=self.bs, num_workers=24)

  def setup(self, stage = None):
      self.train_ds = SinDataset()

#%%

model = HWModel()
model.setup()
#x = model.train_ds[0][0]
#model(torch.unsqueeze(model.train_ds[0][0], 0), model.get_new_hidden(1))
#model.generate_unconditionally(prompt=x, noise_variance=.01)



#%%

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="sin_generation-lightning")

lr_monitor = LearningRateMonitor(logging_interval='step')

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        x = model.train_ds[0][0]

        for noise_variance in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
          sample = pl_module.generate_unconditionally(noise_variance=noise_variance, prompt=x)
          f = plot_stroke(x, sample)
          logger.experiment.add_figure(f'generate with untrained model (noise variance = {noise_variance})', f, 0)


trainer = pl.Trainer(
        max_epochs=args["epochs"],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[MyPrintingCallback()],
        )

logger.log_hyperparams(args)

trainer.fit(model)    

trainer

#%%
trainer = pl.Trainer(
        auto_lr_find=True,
        max_epochs=args["epochs"],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        )
lr_finder = trainer.tuner.lr_find(model, min_lr=1e-3)
print(lr_finder.results)
fig = lr_finder.plot(suggest=True)
fig.show()



# %%

x, y = model.train_ds[0]
print(x)
with torch.no_grad():
    yhat= model(torch.unsqueeze(x, 0), model.get_new_hidden(1))[0]
y, yhat
# %%
x = model.train_ds[0][0]

prompt= x
prediction = model.generate_unconditionally(prompt=prompt, noise_variance=.01)

#print(x, prediction)
plot_stroke(prompt, prediction)



# %%
