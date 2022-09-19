# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""
#%%
3+4

# run 3: max_seq_len: 12, seq_chop_len: 12, max_training: 1, max_validation: 1 epochs: 80
# loss -1.3

# run 4: max_seq_len: 12, seq_chop_len: 12, max_training: 1, max_validation: 1 epochs: 40, tbtt: 60
# loss 0.527 (but loss not calculated right for tbtt)

# run 5:  disable tbtt
# loss 0.42

# run 13: redo calc of loss with tbtt
# loss .4937

# run 14: re-enable tbtt
# loss .50

# run 18: set seq_chop_len to 300
# loss 1.4

# run 18: set tbtt from 60 to 20
# loss 1.4

# run 20: turn tbptt off
# loss 1.4

# run 21: max_training_samples: 1->1000, max_validation_samples: 1->1000
# loss 0.7
# run 21: max_training_samples: 100000, max_validation_samples: 1->1000
# loss -0.3
# run 22: max_training_samples: 100000, max_validation_samples: 1->1000
# loss -0.3

# run 23: lr:.001->.003
# loss -0.4

# Run 24: rnn_hidden_size 200->900
# loss -0.47

# Run 25: epochs 40->120
# loss -0.75
# Run 30: disable tbtt false
# loss -0.9
# Run 40: sort by seq len and do chopping within a batch ("max_seq_len"-> 1000,
#  "seq_chop_len"-> None,
# loss -1.048   Beginning to overfit halfway 
#%%

import matplotlib.pyplot as pyplot
nfrom pathlib import Path
import math
import random
from pytorch_lightning.callbacks import Callback



hwdir = Path('/data') / 'neil' / 'hw'

args = {
    # for model
    'epochs':120,
    'batch_size': 400,
    'lr': .003,
    "rnn_hidden_size": 900,
    "tbptt": 20,
    "disable_tbptt": False,
    "max_seq_len": 1000,
    "seq_chop_len": None,
    "max_training_samples": 100000,
    "max_validation_samples": 1000,
    'num_components': 1,

    # for generation
    "generated_length": 150,
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
  def __init__(self, strokes, stats, chop_length=args['seq_chop_len']):
      """data is a numpy array containing different seq*3 arrays"""
      #print(f'SeqDataSet.__init__: data.shape: {data.shape}')
      self.stats = stats
      self.data = []
      strokes = sorted(strokes, key=lambda a: a.shape[0])
      for stroke in strokes:
        stroke[:, 1:] = stroke[:, 1:] - self.stats['mean'] / self.stats['std']
        if chop_length is not None:
            for i in range(min(stroke.shape[0], args['max_seq_len']) // chop_length):
                self.data.append(stroke[i*chop_length:(i+1)*chop_length])
        else:
            self.data.append(stroke[:args['max_seq_len']])

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

    train_strokes = strokes[:5]

    def calc_stats(values):
        totals = None
        meanUps = np.concatenate([stroke[:, 0] for stroke in values]).mean()
        shapes = [stroke[:, 1:] for stroke in values]
        totals = np.concatenate(shapes)
        return {'mean': totals.mean(axis=0), 'std': totals.std(axis=0), 'meanUps':meanUps}

    stats = calc_stats(train_strokes)
    #print(f'stats: {stats}')

    train_ds = SeqDataset(train_strokes, stats)
    print(f'length: {len(train_ds)}')

    first = train_ds[0][0]
    first[:,1:] = (first[:,1:] * stats['std']) + stats['mean']
    prompt = first[:10,:]
    last=first[10:,:]
    generated=np.copy(last)
    generated[:,1:] = generated[:,1:] + (np.random.rand(generated.shape[0], 2)-0.5)*3
    plot_stroke(generated, prompt=prompt, remainder=last)
    for i in range(len(train_ds)):
        print(i, train_ds[i][0].shape[0])

#%%

import string

class HWModel(pl.LightningModule):
  def __init__(self, lr=args['lr'], bs = args['batch_size']):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 5) 

    self.learning_rate = lr
    self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
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

    #print(f'mean.shape: {mean.shape}, stddev.shape: {stddev.shape}, y.shape: {y.shape}')
    losses = ((mean - y)**2)/(2*stddev**2)  + torch.log(stddev)
    return losses

  def loss(self, y_hat, y):
      #print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
      #print(f'y_hat: [{y_hat[:,:20:]}, {y[:,:20,:]}')
      xy_loss = self.xy_loss(y_hat[:,:,1:], y[:,:,1:]) 
      bce_loss = self.bceWithLogitsLoss(y_hat[:,:,:1], y[:,:,:1])
      #print(f'xy_loss: {xy_loss}, bce_loss = {bce_loss}')
      return xy_loss + bce_loss

  def training_step(self, batch, batch_idx, hiddens=None):
    data, y = batch
    bs, seq_len, _ = data.shape
    #print(f'batch: {bs}, {seq_len}')
    seq_start = 0
    tbptt = args['tbptt']
    hiddens = self.getNewHidden(batch_size=data.shape[0])
    sub_batches = 0
    losses = []
    while seq_start < seq_len:
        sub_batch = (data[:,seq_start:seq_start + tbptt, :],
                     y[:, seq_start:seq_start + tbptt, :])
        #print(f'sub_batch: {(sub_batch[0].shape, sub_batch[1].shape)}')
        
        #print(f"data.shape {data.shape}")
        #print(f"y.shape {y.shape}")
        #print(f'training_step(data.shape={data.shape}, batch_idx={batch_idx}')
        if args['disable_tbptt']:
            hiddens = self.getNewHidden(batch_size=data.shape[0])
        y_hat, hiddens = self(sub_batch[0], hiddens)
        hiddens.detach()
        losses.append(self.loss(y_hat, sub_batch[1]))
        #print(f'loss: {loss}')
        seq_start += tbptt
        sub_batches += 1
    loss = torch.mean(torch.cat(losses, dim=1))
    self.log('loss_train', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    #print(f'validation_step (batch={batch}, batch_idx={batch_idx}')
    data, y = batch
    hidden = self.getNewHidden(batch_size=data.shape[0])
    y_hat, hidden = model(data, hidden)
    #c, t = self.char_accuracy(y_hat, y)
    loss = torch.mean(self.loss(y_hat, y))
    self.log("loss_val", loss)
    #self.logger.experiment.add_scalars("losses", {"val_loss": losses["total"]})
    #self.log('train_loss', losses, prog_bar=True)
    #self.log("val_accuracy", 100. * c / t)
    if batch_idx == 0 and (self.current_epoch % 2) == 0:
      prompt=self.train_ds[0][0][:30,:]
      remainder=self.train_ds[0][0][30:,:]
      sample = self.generate_unconditionally(prompt)
      #print(f'sample: {sample}')
      print(f'generated HW epoch: {self.current_epoch}')
      f = plot_stroke(sample, prompt=prompt, remainder=None)
      self.logger.experiment.add_figure('generated HW', f, self.current_epoch)
    return loss



  def generate_unconditionally(self, prompt=None, output_length=args["generated_length"],
        verbose=False):
    hidden = self.getNewHidden(batch_size=1)
    output = torch.zeros((output_length+1, 3), device=self.device)

    def sample_from_prediction(prediction):
        #print(f'sample_from_prediction: {prediction}')
        result = torch.zeros((3))
        result[0] = 1 if  torch.sigmoid(prediction[0]) > random.random() else 0
        # generate random values with given means and standard devs  
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
        predictions, hidden = self.forward(input, hidden)

      # Only use the last prediction.
      output[idx+1] = sample_from_prediction(predictions[0, -1, :])


    if prompt is not None:
        #skip first (zero) element
        output = output[1:,:]
    #output[:,0] = torch.sigmoid(output[:,0])
    asNumpy = output.cpu().numpy()
    # denormalize:
    asNumpy[:,1:] = asNumpy[:,1:] * self.stats['std'] + self.stats['mean']

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


  @staticmethod
  def dataloader_collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    # Must have same seq_len for each item in batch
    # So, chop them off
    # TODO(neil): Pad short, rather than cut long
    chop_len = min([x.shape[0] for x in xs])
    xs = [x[:chop_len] for x in xs]
    ys = [y[:chop_len] for y in ys]
    result = (torch.stack(xs, dim=0), torch.stack(ys, dim=0))
    return result
              
  def train_dataloader(self):
      return DataLoader(self.train_ds, shuffle=True, batch_size=self.bs, collate_fn=self.dataloader_collate_fn, num_workers=2)

  def val_dataloader(self):
      return DataLoader(self.val_ds, shuffle=False, batch_size=self.bs, collate_fn=self.dataloader_collate_fn, num_workers=2)


  def setup(self, stage = None):
      train_split = int(len(self.strokes)*0.95)
      #valid_split = len(self.strokes) - train_split
      #train_strokes, valid_strokes = random_split(self.strokes, [train_split, valid_split])

      train_strokes = self.strokes[:min(args['max_training_samples'], train_split)]
      valid_strokes = self.strokes[train_split:train_split + args['max_validation_samples']]

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
        #logger.experiment.add_figure('original training image', f, 0)

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
