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
    'batch_size': 512,
    'test_batch_size': 128,
    'epochs':30,
    'truncated_bptt_steps': 500,
    'lr': .001,
    'max_lr': .01,
    'optimizer': 'adam',
    "rnn_hidden_size": 100,
    "rnn_type": "lstm",
    "noise_variance": 0.05,

    # for generation
    'temperature': 0.9,
    "prompt": "A",
    "generated_length": 500,

    # meta
    'seed': 1,
    'log_interval': 1000,

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


import string

class HWModel(pl.LightningModule):
  def __init__(self, lr=.01):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 3) 

    # This is passing back the hiddens but not breaking apart the sequence
    # (e.g., not calling tbptt_split_batch)
    self.truncated_bptt_steps = args['truncated_bptt_steps']

    self.learning_rate = lr
    self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

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
    
  def char_accuracy(self, output, target):
    mostLikely = torch.argmax(output, dim=2)
    #rint(f"mostLikely size: {mostLikely.size()}")
    #print(f"target size: {target.size()}")
    eq = mostLikely.eq(target.view_as(mostLikely))
    #print(f"eq: {eq.size()}, {eq}")
    #print(f"eq.sum(): {eq.sum().size()}, {eq.sum()}")
    correct = eq.sum().item()
    total = torch.numel(eq)
    #print(f"correct, total: {correct}, {total}")
    return correct, total

  def loss(self, y_hat, y):
     # print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
      #print(f'y_hat: [{y_hat[:,:20:]}, {y[:,:20,:]}')
      mse_loss = F.mse_loss(y_hat[:,:,1:], y[:,:,1:]) 
      bce_loss = self.bceWithLogitsLoss(y_hat[:,:,:1], y[:,:,:1])
      #print(f'mse_loss: {mse_loss}, bce_loss = {bce_loss}')
      return mse_loss + bce_loss

  def training_step(self, batch, batch_idx, hiddens):
    #print(f'training_step(batch={batch}, batch_idx={batch_idx}')
    data, y = batch
    #print(f"data.shape {data.shape}")
    y_hat, hiddens = self(data, hiddens)
      
    #correct, total = self.char_accuracy(y_hat, y)

    loss = self.loss(y_hat, y)
    if batch_idx == 0:
        #print(f'epoch: {self.current_epoch} batch: {batch_idx} training loss({y_hat}, {y})')
        self.log('train_loss', loss, prog_bar=True)
        #if self.current_epoch == 0:
            #print(f"saving original training image shape: {y[0].shape}")
            #f = plot_stroke(y[0].cpu().numpy())
            #self.logger.experiment.add_figure('original training image', f, self.current_epoch)
    #self.log('train_accuracy', 100.*correct/total)

    return {"loss": loss, "hiddens": hiddens}

  def validation_step(self, batch, batch_idx):
    #print(f'validation_step (batch={batch}, batch_idx={batch_idx}')
    data, y = batch
    hidden = self.getNewHidden(batch_size=data.shape[0])
    y_hat, hidden = model(data, hidden)
    #c, t = self.char_accuracy(y_hat, y)
    loss = self.loss(y_hat, y)
    self.log("val_loss", loss)
    #self.log("val_accuracy", 100. * c / t)
    if batch_idx == 0:
      sample = self.generateUnconditionally()
      #print(f'sample: {sample}')
      f = plot_stroke(sample)
      self.logger.experiment.add_figure('sample as figure', f, self.current_epoch)
      #self.logger.experiment.add_text('sample as tensor', str(sample.tolist()), self.current_epoch)


  def generateUnconditionally(self, output_length=args["generated_length"], noise_variance=args["noise_variance"]):
    VERBOSE=False
    result = ""
    hidden = self.getNewHidden(batch_size=1)
    output = torch.zeros((output_length+1, 3), device=self.device)
    noise = torch.randn((output_length+1, 2), device=self.device) * noise_variance

    for idx in range(output_length):
      with torch.no_grad():
        input = torch.reshape(output[idx], (1, 1, 3))
        #print(f"generate: input for step {idx}: {input}")
        #print(f"generate: hidden: {hidden}")
        predictions, hidden = self.forward(torch.reshape(output[idx], (1, 1, 3)), hidden)

      # Only use the last prediction.
      output[idx+1, :] = predictions[0, -1, :]

      # Do sampling from the prediction:
      output[:,1:] = output[:,1:] + noise[idx]

      #print(f'output[{idx+1}] = {output[idx+1]}')

    # skip first (zero) element
    asNumpy = output[1:,:].cpu().numpy()

    asNumpy[:, 0] = np.where(asNumpy[:, 0] > 0, 1, 0)
    #print(f'final result: {asNumpy}')
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

from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np

# TODO(neil): make this random sequences rather than fixed?
MAX_SEQ=200
class SeqDataset(Dataset):
  def __init__(self, data):
      """data is a numpy array containing different seq*3 arrays"""
      #print(f'SeqDataSet.__init__: data.shape: {data.shape}')
      self.data = data[:1][:MAX_SEQ+1]

  def __len__(self):
    #print(f'SeqDataSet len(self.data)={len(self.data)} self.seq_length={self.seq_length}')
    return len(self.data)

  def __getitem__(self, idx):
    t = torch.as_tensor(self.data[idx])
    #print(f'SeqDataset.getitem({idx}): returning shape:{t.shape}')
    return t[:-1], t[1:]

class HWDataModule(pl.LightningDataModule): 
    def __init__(self, bs = 1):

        super().__init__()
        self.bs = bs

    def prepare_data(self):
        datadir = hwdir / 'data'
        strokes = np.load(datadir / 'strokes-py3.npy', allow_pickle=True)
        import string
        self.dataset1 = SeqDataset(strokes[0:1])
        self.dataset2 = SeqDataset(strokes[1:2])

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=False, batch_size=self.bs, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.bs, num_workers=24)


    def setup(self, stage = None):
        #train_split = int(len(self.dataset)*0.5)
        #valid_split = len(self.dataset) - train_split
        #self.train_ds, self.val_ds = random_split(self.dataset, [train_split, valid_split])
        self.train_ds = self.dataset1   
        self.val_ds = self.dataset2   
        #print(f'HWDataModule: len(train_ds) = {len(self.train_ds)}, len(val_ds)={len(self.val_ds)}')



model = HWModel()

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="hw_generation-lightning")

lr_monitor = LearningRateMonitor(logging_interval='step')

data_module = HWDataModule()
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        dl = data_module.train_dataloader()
        print(f"dl={dl}")
        f = plot_stroke(dl.dataset[0][0].numpy())
        print(f"size of handwriting: {dl.dataset[0][0].numpy().shape}")
        logger.experiment.add_figure('original training image', f, 0)

        sample = pl_module.generateUnconditionally()
        #print(f'sample: {sample}')
        f = plot_stroke(sample)
        logger.experiment.add_figure('generate before training', f, 0)
        #self.logger.experiment.add_text('sample as tensor', str(sample.tolist()), self.current_epoch)



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

trainer.fit(model, datamodule=data_module)    
"""
def generateUnconditionally():
    VERBOSE=False
    strokeCount=0
    strokeToCopy= SeqDataset(train_samples)[0]

    def resetModel():
      nonlocal strokeCount
      #model.reset_states()
      strokeCount=0

    def runModel(strokes):
      # should take the strokes so far and return a single model output
      # np array of size NUM_OUTPUTS

      if not useFirstTrainingSample:
        # Run the model.
        # Input.shape is [batch, sequence_length, input_per_sequence]
        inputs=torch.reshape(torch.tensor(strokes), (1, len(strokes), 3)).to(device)
        with torch.no_grad():
          predictions = model(inputs)
        #print('predictions.shape', predictions.shape)
        # Only use the last prediction.
        prediction = predictions[0, -1, :].cpu().numpy()
        #print('prediction.shape', prediction.shape)
        #print('prediction', prediction)
        return prediction
      else:
        # Predict to be like trainX[0]
        nonlocal strokeCount
        global trainX
        sample = trainX[0][strokeCount]
        strokeCount = strokeCount + 1
        return np.array([
                        sample[0], # penUp
                        1.0, # weight
                        sample[1], # x mean
                        sample[2], # y mean
                        0.1, # x stddev
                        0.1, # y stddev
                        0.3, # correlation
                       ])

    def generateOneStep(prediction):
      # prediction is of shape NUM_OUTPUTS
      result = np.zeros((3))
      nextIndex=1
      # We calculate based on prediction so that we can run on saved
      # predictions/models that used a different value for NUM_MIXTURE_COMPONENTS
      numMixtureComponents = prediction.shape[0] // FEATURES_PER_MIXTURE
      #print('numMixtureComponents', numMixtureComponents)
      # weights
      weights = prediction[nextIndex:nextIndex+numMixtureComponents]
      nextIndex += numMixtureComponents
      #print('weights.shape', weights.shape)
      #print('weights', weights)

      # means
      hmeans = prediction[nextIndex:nextIndex + numMixtureComponents]
      nextIndex += numMixtureComponents
      vmeans = prediction[nextIndex:nextIndex + numMixtureComponents]
      nextIndex += numMixtureComponents
      #print('hmeans', hmeans)
      #print('vmeans', vmeans)
      means = np.transpose(np.stack([hmeans, vmeans]))
      #print('means.shape', means.shape)


      # standard deviations
      hstddevs = prediction[nextIndex:nextIndex + numMixtureComponents]
      nextIndex += numMixtureComponents
      vstddevs = prediction[nextIndex:nextIndex + numMixtureComponents]
      nextIndex += numMixtureComponents
      #print('hstddevs', hstddevs)
      #print('vstddevs', vstddevs)

      # correlations
      rhos = prediction[nextIndex:nextIndex + numMixtureComponents*CORRELATIONS_PER_MIXTURE]
      #print('rhos', rhos)
      #print('hstddevs*hstddevs', hstddevs*hstddevs)
      #print('hstddevs*vstddevs*rhos', hstddevs*vstddevs * rhos)
      cov = np.array([[hstddevs * hstddevs, hstddevs * vstddevs * rhos],
                [hstddevs * vstddevs * rhos, vstddevs * vstddevs]]).transpose((2, 1, 0))

      #print('cov.shape', cov.shape)
      #print('cov', cov)
      #print('means[0]', means[0])
      #print('cov[0]', cov[0])

      draws = np.array([np.random.multivariate_normal(means[i], cov[i]) for i in range(numMixtureComponents)])
      #print(f'draws = {draws}')
      weights = np.reshape(weights, (numMixtureComponents, 1))
      #print(f'weights = {weights}')
      weightedDraws = weights*draws
      #print('weightedDraws', weightedDraws)
      #print('sum(weightedDraws)', np.sum(weightedDraws, axis=0))
      result[1:3] = np.sum(weightedDraws, axis=0)
      # De-normalize
      result[1:3]  = result[1:3] * stats['std'] + stats['mean']
      result[0] = 1 if prediction[0] > 0.5 else 0
      return result.tolist()

    resetModel()
    stroke = np.zeros((3))
    strokes = [[0, 0., 0.]]
    for step in range(180):
      prediction = runModel(strokes)
      #print('prediction', prediction)
      stroke = generateOneStep(prediction)
      #print('stroke', stroke)
      strokes.append(stroke)
    #print(strokes)
    result = np.asarray(strokes[1:])
    result[-1,0] = 1 # penup for last stroke
    #print(result)
    return result
        """
