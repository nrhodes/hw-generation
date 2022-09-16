# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""
#%%
3+4
#%%

import matplotlib.pyplot as pyplot
from pathlib import Path
import math
import random
from pytorch_lightning.callbacks import Callback



hwdir = Path('/data') / 'neil' / 'hw'

args = {
    # for model
    'epochs':17,
    'batch_size': 400,
    'lr': .001,
    'optimizer': 'adam',
    "rnn_hidden_size": 200,
    "rnn_type": "gru",
    "noise_variance": 0.1,

    # for generation
    "generated_length": 100,
}


#%%
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

def plot_stroke(stroke, prompt=None, remainder=None, save_name=None):
    # Plot a single example.
    #print(f'stroke before penup: {stroke}')
    f, ax = pyplot.subplots()

    if torch.is_tensor(stroke):
        stroke = stroke.numpy()
    if torch.is_tensor(prompt):
        prompt = prompt.numpy()
    if torch.is_tensor(remainder):
        remainder = remainder.numpy()

    first_x = 0
    first_y = 0
    if prompt is not None:
        x_prompt = np.cumsum(prompt[:, 1])
        y_prompt = np.cumsum(prompt[:, 2])
        ax.plot(x_prompt, y_prompt, 'b-', linewidth=1)
        first_x = x_prompt[-1]
        first_y = y_prompt[-1]

    x = np.cumsum(stroke[:, 1]) + first_x
    y = np.cumsum(stroke[:, 2]) + first_y

    #print('entire stroke', x, y)
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(7., 4.)

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

    if remainder is not None:
        x_rem = np.cumsum(remainder[:, 1]) + first_x
        y_rem = np.cumsum(remainder[:, 2]) + first_y
        #print(f'x_rem', x_rem)
        #print(f'y_rem', y_rem)
        ax.plot(x_rem, y_rem, 'g-', linewidth=1)

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

#%%
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np

class SeqDataset(Dataset):
  def __init__(self, strokes, stats, chop_length=60):
      """data is a numpy array containing different seq*3 arrays"""
      #print(f'SeqDataSet.__init__: data.shape: {data.shape}')
      self.stats = stats
      self.data = []
      for stroke in strokes:
        stroke[:, 1:] = stroke[:, 1:] - self.stats['mean'] / self.stats['std']
        for i in range(stroke.shape[0] // chop_length):
            self.data.append(stroke[i*chop_length:(i+1)*chop_length])

  def __len__(self):
    #print(f'SeqDataSet len(self.data)={len(self.data)} self.seq_length={self.seq_length}')
    return len(self.data)

  def __getitem__(self, idx):
    t = torch.as_tensor(self.data[idx])
    return t[:-1], t[1:]

#%%
3+4
if True:
    datadir = hwdir / 'data'
    strokes = np.load(datadir / 'strokes-py3.npy', allow_pickle=True)
    strokes[0].shape[0]

    train_strokes = strokes[:1]

    def calc_stats(values):
        totals = None
        meanUps = np.concatenate([stroke[:, 0] for stroke in values]).mean()
        shapes = [stroke[:, 1:] for stroke in values]
        totals = np.concatenate(shapes)
        return {'mean': totals.mean(axis=0), 'std': totals.std(axis=0), 'meanUps':meanUps}

    stats = calc_stats(train_strokes)
    #print(f'stats: {stats}')

    train_ds = SeqDataset(train_strokes, stats)

    first = train_ds[0][0]
    first[:,1:] = (first[:,1:] * stats['std']) + stats['mean']
    prompt = first[:30,:]
    last=first[30:,:]
    generated=np.copy(last)
    generated[:,1:] = generated[:,1:] + (np.random.rand(generated.shape[0], 2)-0.5)*3
    plot_stroke(generated, prompt=prompt, remainder=last)

#%%

import string

class HWModel(pl.LightningModule):
  def __init__(self, lr=args['lr'], bs = args['batch_size']):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 5) 

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
    x[:,:,3:] = torch.exp(x[:,:,3:])   # convert to non-negative std dev
    return x, hidden
    
  def xy_loss(self, yhat, y):
    mean = yhat[:,:,0:2]
    stddev = yhat[:,:,2:4]
    losses = ((mean - y)**2)/(2*stddev**2)  + torch.log(stddev)
    return torch.mean(losses)

  def loss(self, y_hat, y):
      #print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
      #print(f'y_hat: [{y_hat[:,:20:]}, {y[:,:20,:]}')
      xy_loss = self.xy_loss(y_hat[:,:,1:], y[:,:,1:]) 
      bce_loss = self.bceWithLogitsLoss(y_hat[:,:,:1], y[:,:,:1])
      #print(f'mse_loss: {mse_loss}, bce_loss = {bce_loss}')
      return {
              'total': xy_loss + bce_loss,
              'xy': xy_loss,
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
    self.log('loss_train', losses, prog_bar=True)
    if batch_idx == 0:
        pass
        #print(f'epoch: {self.current_epoch} batch: {batch_idx} training loss({y_hat}, {y})')
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
    #self.log("loss_val", losses)
    self.logger.experiment.add_scalars("losses", {"val_loss": losses["total"]})
    #self.log('train_loss', losses, prog_bar=True)
    #self.log("val_accuracy", 100. * c / t)
    if batch_idx == 0:
      prompt=self.train_ds[0][0][:30,:]
      remainder=self.train_ds[0][0][30:,:]
      sample = self.generate_unconditionally(prompt)
      #print(f'sample: {sample}')
      print(f'generated HW epoch: {self.current_epoch}')
      f = plot_stroke(sample, prompt=prompt, remainder=remainder)
      self.logger.experiment.add_figure('generated HW', f, self.current_epoch)
      #self.logger.experiment.add_text('sample as tensor', str(sample.tolist()), self.current_epoch)
    return {'loss': losses['total']}



  def generate_unconditionally(self, prompt=None, output_length=args["generated_length"], noise_variance=args["noise_variance"],
        verbose=False):
    hidden = self.getNewHidden(batch_size=1)
    output = torch.zeros((output_length+1, 3), device=self.device)
    #print(f"noise: {noise}")

    def sample_from_prediction(prediction):
        #print(f'sample_from_prediction: {prediction}')
        result = torch.zeros((3))
        result[0] = 1 if  torch.sigmoid(prediction[0]) > random.random() else 0
        # generate andom values with given means and standard devs  
        result[1:] = torch.normal(prediction[1:3], prediction[3:5])
        return result


    if prompt is not None:
        prompt = prompt.type_as(hidden)
        with torch.no_grad():
            input = torch.unsqueeze(prompt, 0)
        if verbose:
            print(f"prompt: input to forward: {input}")
        predictions, hidden = self.forward(input, hidden)
        if verbose:
            print(f"predictions: {predictions}")
        output[0] = sample_from_prediction(predictions[0, -1, :])

    for idx in range(output_length):
      with torch.no_grad():
        input = torch.reshape(output[idx], (1, 1, 3))
        predictions, hidden = self.forward(torch.reshape(output[idx], (1, 1, 3)), hidden)

      # Only use the last prediction.
      output[idx+1, :] = sample_from_prediction(predictions[0, -1, :])


    if prompt is not None:
        #skip first (zero) element
        output = output[1:,:]
    # convert to probabilities
    #output[:,0] = torch.sigmoid(output[:,0])
    asNumpy = output.cpu().numpy()
    # denormalize:
    asNumpy[:,1:] = asNumpy[:,1:] * self.stats['std'] + self.stats['mean']

    # Sample whether penUp or penDown based on probability of penUp
   # asNumpy[:, 0] = np.where(asNumpy[:, 0] > np.random.rand(asNumpy.shape[0]), 1, 0)
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
      self.strokes = strokes

  def train_dataloader(self):
      return DataLoader(self.train_ds, shuffle=False, batch_size=self.bs, num_workers=24)

  def val_dataloader(self):
      return DataLoader(self.val_ds, shuffle=False, batch_size=self.bs, num_workers=24)


  def setup(self, stage = None):
      train_split = int(len(self.strokes)*0.9)
      valid_split = len(self.strokes) - train_split
      train_strokes, valid_strokes = random_split(self.strokes, [train_split, valid_split])

      #train_strokes = self.strokes[:1]
      #valid_strokes = self.strokes[1:2]
      def calc_stats(values):
        totals = None
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
        print(f'original training image')
        f = plot_stroke(asNumpy)
        #print(f"size of handwriting: {dl.dataset[0][0].numpy().shape}")
        logger.experiment.add_figure('original training image', f, 0)

        for noise_variance in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]: #
          sample = pl_module.generate_unconditionally(noise_variance=noise_variance)
          print(f'generated from untrained model (noise={noise_variance})')
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

# %%
