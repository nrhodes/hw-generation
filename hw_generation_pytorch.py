# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""

import io
import matplotlib.pyplot as pyplot
from pathlib import Path
from PIL import Image
from torchvision import transforms


import requests
hwdir = Path('/data') / 'neil' / 'hw'

args = {
    # for model
    'batch_size': 512,
    'test_batch_size': 128,
    'epochs':20,
    'max_seq_length': 50,
    'truncated_bptt_steps': 5,
    'lr': .001,
    'max_lr': .01,
    'steps_per_epoch': 10,
    'optimizer': 'adam',
    "rnn_hidden_size": 100,
    "rnn_type": "lstm",

    # for generation
    'temperature': 0.9,
    "prompt": "A",
    "generated_length": 50,

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

torch.manual_seed(args["seed"])
use_cuda = torch.cuda.is_available()
print("cuda is available", use_cuda)
device = torch.device("cuda")




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Sampler


import string

# TODO: make these not be globals

class HWModel(pl.LightningModule):
  def __init__(self, lr=.01):
    super().__init__()
    self.hidden_size = args["rnn_hidden_size"]
    self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 3) 
    self.truncated_bptt_steps = args['truncated_bptt_steps']
    self.learning_rate = lr
    self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    # Return the hidden tensor(s) to pass to forward
  def getNewHidden(self, batch_size):
    return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
           torch.zeros(1, batch_size, self.hidden_size,device=self.device))

  def forward(self, x, hidden):
    VERBOSE=False
    if VERBOSE:
      print(f'Forward: size of input: {x.size()}')

    (x, hidden) = self.lstm(x, hidden)

    if VERBOSE:
      print(f'Forward: size after rnn: {x.size()}')
      #print(f'Forward: after rnn: {x} ')
    x = self.fc(x)
    if VERBOSE:
      print(f'Forward: size after fc: {x.size()}')
      #print(f'Forward:  after fc: {x}')
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
      return (F.mse_loss(y_hat[1:], y[1:]) +
        self.bceWithLogitsLoss(y_hat[:1], y[:1]))

  def training_step(self, batch, batch_idx, hiddens):
    data, y = batch
    #print(f"data.shape {data.shape}")
    y_hat, hiddens = self(data, hiddens)
      
    #correct, total = self.char_accuracy(y_hat, y)

    loss = self.loss(y_hat, y)
    self.log('train_loss', loss, prog_bar=True)
    #self.log('train_accuracy', 100.*correct/total)

    return {"loss": loss, "hiddens": hiddens}

  def validation_step(self, batch, batch_idx):
    print('validation_step: begin')
    def plot_to_image(figure):
      """Converts the matplotlib plot specified by 'figure' to a PNG image and
      returns it. The supplied figure is closed and inaccessible after this call."""
      # Save the plot to a PNG in memory.
      buf = io.BytesIO()
      pyplot.savefig(buf, format='png')
      # Closing the figure prevents it from being displayed directly inside
      # the notebook.
      pyplot.close(figure)
      buf.seek(0)
      # Convert PNG buffer to TF image
      image = Image.open(buf)
      #print(f'image={image}')
      # Add the batch dimension
      return transforms.ToTensor()(image)

    def plot_stroke(stroke, save_name=None):
      # Plot a single example.
      f, ax = pyplot.subplots()

      x = np.cumsum(stroke[:, 1])
      y = np.cumsum(stroke[:, 2])
  
      size_x = x.max() - x.min() + 1.
      size_y = y.max() - y.min() + 1.

      f.set_size_inches(5. * size_x / size_y, 5.)

      cuts = np.where(stroke[:, 0] == 1)[0]
      start = 0

      for cut_value in cuts:
          ax.plot(x[start:cut_value], y[start:cut_value],
                  'k-', linewidth=3)
          start = cut_value + 1
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

    print(f'validation_step')
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
      t = plot_to_image(f)
      print(f'image_tensor.shape: {t.shape}')

      self.logger.experiment.add_image('sample', t, self.current_epoch)


  def generateUnconditionally(self, output_length=args["generated_length"]):
    VERBOSE=False
    result = ""
    hidden = self.getNewHidden(batch_size=1)
    output = torch.zeros((output_length+1, 3), device=self.device)
    print(f'generate: output.shape: {output.shape}')

    for step, idx in enumerate(range(output_length)):
      with torch.no_grad():
        predictions, hidden = self.forward(torch.reshape(output[idx], (1, 1, 3)), hidden)
      #print('predictions.shape', predictions.shape)
      # Only use the last prediction.
      output[idx, :] = predictions[0, -1, :]
      output[idx,0] = 1 if predictions[0, -1, 0] > 0 else 0

    output[-1,0] = 1  # final penup
    # skip first element
    asNumpy = output[1:,:].cpu().numpy()
    return asNumpy

  def configure_optimizers(self):
    print('self.learning_rate', self.learning_rate)
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

# Break up given data into chunks of max_seq_length each.
# TODO(neil): make this random sequences rather than fixed
class SeqDataset(Dataset):
  def __init__(self, data, seq_length=args['max_seq_length']):
      """data is a numpy array containing different seq*3 arrays"""
      #self.seq_length = seq_length
      #self.num_sequences = (len(data) - self.seq_length) // self.seq_length
      self.data = data

  def __len__(self):
    return 2
    #return (len(self.data) - self.seq_length) // self.seq_length

  def __getitem__(self, idx):
    t = torch.as_tensor(self.data[idx])
    #print(t)
    return t[:-1], t[1:]

class HWDataModule(pl.LightningDataModule): 
    def __init__(self, bs = 1):

        super().__init__()
        self.bs = bs

    def prepare_data(self):
        datadir = hwdir / 'data'
        strokes = np.load(datadir / 'strokes-py3.npy', allow_pickle=True)
        import string
        self.dataset = SeqDataset(strokes)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs)


    def setup(self, stage = None):
        train_split = int(len(self.dataset)*0.5)
        valid_split = len(self.dataset) - train_split
        self.train_ds, self.val_ds = random_split(self.dataset, [train_split, valid_split])



model = HWModel()

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="hw_generation-lightning")

lr_monitor = LearningRateMonitor(logging_interval='step')

data_module = HWDataModule()
trainer = pl.Trainer(progress_bar_refresh_rate=1,
        max_epochs=args["epochs"],
        gpus=1,
        logger=logger,
        callbacks=[lr_monitor],
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
