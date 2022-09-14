# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""

import io
import matplotlib.pyplot as pyplot
from pathlib import Path
from PIL import Image
from pytorch_lightning.callbacks import Callback

from torchvision import transforms


import requests
hwdir = Path('/data') / 'neil' / 'hw'

args = {
    # for model
    'epochs':1,
    'batch_size': 700,
    'test_batch_size': 1024,
    'training_set_size': 1000,
    'validation_set_size': 1,
    'lr': .001,
    'max_lr': .01,
    'optimizer': 'adam',
    "rnn_hidden_size": 200,
    "rnn_type": "gru",
    "noise_variance": 0.1,

    # for generation
    "generated_length": 100,
}


"""# Initialization"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

#torch.manual_seed(args["seed"])
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Sampler

def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    #print(f'stroke before penup: {stroke}')
    f, ax = pyplot.subplots()

    #stroke[-1,0] = 1   #penup so we can see the last part of the stroke
    #print(f'stroke after penup: {stroke}')

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    #print('entire stroke', x, y)
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    #print(f'np.where(stroke[:, 0] == 1): {np.where(stroke[:, 0] == 1)}')
    #print(f'plot_stroke: cuts={cuts}')
    start = 0

    for cut_value in cuts:
        #print('black', start, cut_value)
        #print(x[start:cut_value], y[start:cut_value])
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        # show pen up part in red
        if cut_value + 2 < len(y):
            #print('red', cut_value-1, cut_value+2)
            ax.plot(x[cut_value-1:cut_value+2], y[cut_value-1:cut_value+2],
                    'r-', linewidth=2)
        start = cut_value + 1

    # final stroke. Especially important if there were no penups specified
    last_cut = cuts[-1] if len(cuts) > 0 else 0
    ax.plot(x[last_cut:len(x)], y[last_cut:len(y)],
            'r-', linewidth=2)

    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
      try:
        pyplot.savefig(
            save_name,
            bbox_inches='tight',
            pad_inches=0.5)
      except Exception:
        print("Error building image!: " + save_name)

    return f


from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np

# TODO(neil): make this random sequences rather than fixed?
MAX_SEQ=300
class SeqDataset(Dataset):
  def __init__(self, data, stats):
      """data is a numpy array containing different seq*3 arrays"""
      #print(f'SeqDataSet.__init__: data.shape: {data.shape}')
      self.data = data
      self.stats = stats

  def __len__(self):
    #print(f'SeqDataSet len(self.data)={len(self.data)} self.seq_length={self.seq_length}')
    return len(self.data)

  def __getitem__(self, idx):
    asNumpy = self.data[idx][:MAX_SEQ]
    asNumpy[:,1:] = (asNumpy[:,1:] - self.stats['mean']) / self.stats['std']
    t = torch.as_tensor(asNumpy)
    #print(f'SeqDataset.getitem({idx}): returning shape:{t.shape}')
    return t[:-1], t[1:]

import string

class HWModel(pl.LightningModule):
  def __init__(self, lr=args['lr'], bs = args['batch_size']):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 3) 

    self.learning_rate = lr
    self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
    self.bs = bs

    # Return the hidden tensor(s) to pass to forward
  def getNewHidden(self, batch_size):
    if False:
        # LSTM
        return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
               torch.zeros(1, batch_size, self.hidden_size,device=self.device))
    else:
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

  def forward(self, x, hidden):
    VERBOSE=False
    if VERBOSE:
        print(f'Forward: input: {x}, hidden: {hidden}')

    (x, hidden) = self.rnn(x, hidden)

    if VERBOSE:
      print(f'Forward: size after rnn: {x.size()}')
      print(f'Forward: after rnn: {x} ')
    x = self.fc(x)
    if VERBOSE:
      print(f'Forward: size after fc: {x.size()}')
      print(f'Forward:  after fc: {x}')
    return x, hidden
    
  def loss(self, y_hat, y):
      #print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
      #print(f'y_hat: [{y_hat[:,:20:]}, {y[:,:20,:]}')
      mse_loss = F.mse_loss(y_hat[:,:,1:], y[:,:,1:]) 
      bce_loss = self.bceWithLogitsLoss(y_hat[:,:,:1], y[:,:,:1])
      #print(f'mse_loss: {mse_loss}, bce_loss = {bce_loss}')
      return {
              'total': mse_loss + bce_loss,
              'xy': mse_loss,
              'penup': bce_loss,
      }

  def training_step(self, batch, batch_idx, hiddens=None):
    data, y = batch
    #print(f"data.shape {data.shape}")
    #print(f"y.shape {y.shape}")
    #print(f'training_step(data.shape={data.shape}, batch_idx={batch_idx}')
    if hiddens is None:
        hiddens = self.getNewHidden(batch_size=data.shape[0])
    y_hat, hiddens = self(data, hiddens)
      
    #correct, total = self.char_accuracy(y_hat, y)
    #print(f"training: beg of y_hat:{y_hat[0,:5,:].detach().cpu().numpy()}")
    #print(f"training: beg of y:{y[0,:5,:].detach().cpu().numpy()}")

    losses = self.loss(y_hat, y)
    if batch_idx == 0:
        #print(f'epoch: {self.current_epoch} batch: {batch_idx} training loss({y_hat}, {y})')
        self.log('loss_train', losses, prog_bar=True)
        #if self.current_epoch == 0:
            #print(f"saving original training image shape: {y[0].shape}")
            #f = plot_stroke(y[0].cpu().numpy())
            #self.logger.experiment.add_figure('original training image', f, self.current_epoch)
    #self.log('train_accuracy', 100.*correct/total)

    self.logger.experiment.add_scalars("losses", {"train_loss": losses["total"]})
    return {'loss': losses['total']}

  def validation_step(self, batch, batch_idx):
    #print(f'validation_step (batch={batch}, batch_idx={batch_idx}')
    data, y = batch
    hidden = self.getNewHidden(batch_size=data.shape[0])
    y_hat, hidden = model(data, hidden)
    #c, t = self.char_accuracy(y_hat, y)
    losses = self.loss(y_hat, y)
    self.log("loss_val", losses)
    self.logger.experiment.add_scalars("losses", {"val_loss": losses["total"]})
    #self.log('train_loss', losses, prog_bar=True)
    #self.log("val_accuracy", 100. * c / t)
    if batch_idx == 0:
      sample = self.generateUnconditionally()
      #print(f'sample: {sample}')
      f = plot_stroke(sample)
      self.logger.experiment.add_figure('generated HW', f, self.current_epoch)
      #self.logger.experiment.add_text('sample as tensor', str(sample.tolist()), self.current_epoch)
    return {'loss': losses['total']}


  def generateUnconditionally(self, output_length=args["generated_length"], noise_variance=args["noise_variance"]):
    VERBOSE=False
    result = ""
    hidden = self.getNewHidden(batch_size=1)
    output = torch.zeros((output_length+1, 3), device=self.device)
    noise = torch.randn((output_length+1, 3), device=self.device) * noise_variance
    #print(f"noise: {noise}")

    for idx in range(output_length):
      with torch.no_grad():
        input = torch.reshape(output[idx], (1, 1, 3))
        predictions, hidden = self.forward(torch.reshape(output[idx], (1, 1, 3)), hidden)

      # Only use the last prediction.
      output[idx+1, :] = predictions[0, -1, :]

      # Do sampling from the prediction:
      output[idx+1] = output[idx+1] + noise[idx]


    #skip first (zero) element
    output = output[1:,:]
    # convert to probabilities
    output[:,0] = torch.sigmoid(output[:,0])
    # skip first (zero) element
    asNumpy = output.cpu().numpy()
    # denormalize:
    asNumpy[:,1:] = asNumpy[:,1:] * self.stats['std'] + self.stats['mean']

    # Sample whether penUp or penDown based on probability of penUp
    asNumpy[:, 0] = np.where(asNumpy[:, 0] > np.random.rand(output_length), 1, 0)
    return asNumpy

  def configure_optimizers(self):
   # print('self.learning_rate', self.learning_rate)
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #optimizer,
        #max_lr=self.learning_rate,
        #total_steps=self.trainer.estimated_stepping_batches
    #)
    return [optimizer], []

  def prepare_data(self):
      datadir = hwdir / 'data'
      strokes = np.load(datadir / 'strokes-py3.npy', allow_pickle=True)
      import string
      TRAINING_SET_SIZE=args['training_set_size']
      VALIDATION_SET_SIZE=args['validation_set_size']
      self.strokes = strokes

  def train_dataloader(self):
      return DataLoader(self.train_ds, shuffle=False, batch_size=self.bs, num_workers=24)

  def val_dataloader(self):
      return DataLoader(self.val_ds, shuffle=False, batch_size=self.bs, num_workers=24)


  def setup(self, stage = None):
      train_split = int(len(self.strokes)*0.9)
      valid_split = len(self.strokes) - train_split
      train_strokes, valid_strokes = random_split(self.strokes, [train_split, valid_split])

      def calc_stats(values):
        totals = None
        upDowns = [stroke[:, 0] for stroke in values]
        meanUps = np.concatenate([stroke[:, 0] for stroke in values]).mean()
        shapes = [stroke[:, 1:] for stroke in values]
        totals = np.concatenate(shapes)
        return {'mean': totals.mean(axis=0), 'std': totals.std(axis=0), 'meanUps':meanUps}

      self.stats = calc_stats(train_strokes)
      print(f'stats: {self.stats}')

      self.train_ds = SeqDataset(train_strokes, self.stats)
      self.val_ds = SeqDataset(valid_strokes, self.stats)
      print(f'HWDataModule: len(train_ds) = {len(self.train_ds)}, len(val_ds)={len(self.val_ds)}')


model = HWModel()

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="hw_generation-lightning")

lr_monitor = LearningRateMonitor(logging_interval='step')

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        #print("Training is starting")
        dl = model.train_dataloader()
        #print(f"dl={dl}")
        asNumpy = dl.dataset[0][0].numpy()
        asNumpy[:,1:] = asNumpy[:,1:] * model.stats['std'] + model.stats['mean']
        f = plot_stroke(asNumpy)
        #print(f"size of handwriting: {dl.dataset[0][0].numpy().shape}")
        logger.experiment.add_figure('original training image', f, 0)

        for noise_variance in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]: #
          sample = pl_module.generateUnconditionally(noise_variance=noise_variance)
          f = plot_stroke(sample)
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
