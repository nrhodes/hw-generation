# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""
#%%

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

# Run 43: Add num_components code. Still run with just one component, though.
# loss -0.95   Random diff from 40?

# Run 44: num_components 1->5
# loss -0.4.

# Run 45. Back to 1 num_components=1  Problem may be in num workers for dataloader?
# loss 

# Run 55. Add old/new code for num_components (verify new code with num_components does same
# calculations as old code does)  Also, num_workers back down to 2 for dataloader
# loss @ 60 epochs: -.8792 vs run 40 @ .8454

# Run 58. 5 components. Switch to 1 worker for dataloader
# loss:  -0.1 (instead of -0.8) (@ 60 epochs)  Crappy!

# Run 59: in forward function, force weight of first component to 1, all others to 0
# Should act like only have 1 component.
# Isn't acting that way. Is following path of Run 58!

# Run 99: Set args['force_to_1_component'] to True. In loss function, calculate
# old and new way.
# loss now is -1.697 (may not be comparable since xy_loss now returns bs X seq_len tensor
# rather than bs X seq_len X 2). However, generated samples look good.

# Run 108: Set args['force_to_1_component'] to False (now have 5 components)
# loss: .01 (big spike at the end)

# Run 123: Compute loss using MultivariateDistribution (with a single component)
# loss will now be on different scale
# loss: .5095

# Run 134: Do sampling MultivariateDistribution (with a single component)
# loss: .504

# Run 141: Add correlation (with a single component)
# loss: .22

# Run 144: Add slight perturbation to tbptt
# loss: .47

# Run 145: Add masking code (but without actually creating -1 penups)
# loss: .4402

# Run 157: Add actual masking: Create -1 penups for short sequences
# loss: 4.2
# Not good!

# Run 158: Go back to shuffle=True for training dataloader and old truncate
# collate_func
# loss: .47

# Run 158: Go back to shuffle=True for training dataloader and old truncate
# collate_func
# loss: .47

# Run ??: Use MixtureSameFamily (with nc still == 1)
# loss: .11

# Run 165: Use MixtureSameFamily (with nc ==20)
# loss: -0.16 (@ end: -0.38)

# Run 166: Weight updown_loss by 3 (also increase batch size)
# loss: -0.01

# Run 167: increase epochs, increase batch size, 
# Loss: -.1094

# Run 172: samples every 5 epochs, no prompt, generated_length: 700->400

# Run 285: Add conditional: slow (16m/batch)
# Loss: 

# Run 287: bs: 2->512 (20s/batch), epochs: 60
# Loss: 

# Run 288: max_seq_len->10000, K->10, 30s/batchq

# Run 316: Rewrote forward (to do one call of rnn layer 2 and 3)
# loss: .2688

# Run 321: Add window weights heatmap
# loss: 

# Run 322: 200 epochs, batch size: 768 19s/batch

# Run 348: Fix kappa (was resetting at every forward)
# loss:0.4

# Run 355: Some fixes to weighting
# loss: 0.92

# Run 384: fix forward to save ws correctly
# loss: .3755

# Run 387: Disable conditional
# loss: 



#%%


import matplotlib.pyplot as pyplot
from pathlib import Path
import random
from pytorch_lightning.callbacks import Callback

hwdir = Path('/data') / 'neil' / 'hw'

#%%
"""# Initialization"""


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D

use_cuda = torch.cuda.is_available()
device = torch.device("cuda")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def run(args):
    # Plot what was predicted (stroke), the prompt, if any,
    # the remainder after the prompt (ground-truth)
    def plot_stroke(stroke, prompt=None, remainder=None, save_name=None):
        # Plot a single example.
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

        size_x = x.max() - x.min() + 1.
        size_y = y.max() - y.min() + 1.

        f.set_size_inches(7., 4.)

        cuts = np.where(stroke[:, 0] == 1)[0]
        start = 0

        for cut_value in cuts:
            ax.plot(x[start:cut_value], y[start:cut_value],
                    'k-', linewidth=3)
            # show pen up part in red
            if cut_value + 2 < len(y):
                #print('red', cut_value-1, cut_value+2)
                ax.plot(x[cut_value-1:cut_value+2], y[cut_value-1:cut_value+2],
                        'r-', linewidth=1)
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

    def plot_heatmap(heatmap):
        fig, ax = pyplot.subplots()
        pos = ax.imshow(heatmap)
        fig.colorbar(pos, ax=ax)
        #c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        #f.colorbar(c, ax=ax)
        return fig


    #%%
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader
    import numpy as np

    CHARS="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "

    class HWPairsDataset(Dataset):

      def __init__(self, strokes, texts, stats):
          self.stats = stats
          strokes_texts = sorted(zip(strokes, texts), key=lambda a: a[0].shape[0])
          strokes = [s for s, _ in strokes_texts]
          texts = [t for _, t in strokes_texts]
          self.setup_texts(texts)
          self.setup_strokes(strokes)

      @staticmethod
      def ord_value(c):
        index = CHARS.find(c)
        if index >=0:
            return index
        else:
            return len(CHARS)

      @staticmethod
      def char_value(i):
        if i < len(CHARS):
            return CHARS[i]
        else:
            return "?"

      @staticmethod
      def to_onehot(text):
        ords = [HWPairsDataset.ord_value(c) for c in text]
        return F.one_hot(torch.tensor(ords), num_classes=len(CHARS)+1)
      
      @staticmethod
      def from_onehot(onehot):
        chars = []
        classes = torch.argmax(onehot, dim=1)
        for i in range(classes.shape[0]):
            chars.append(HWPairsDataset.char_value(classes[i]))
        return "".join(chars)

      def setup_strokes(self, strokes):
        self.strokes = []
        for stroke in strokes:
          stroke[:, 1:] = stroke[:, 1:] - self.stats['mean'] / self.stats['std']
          self.strokes.append(stroke[:args.max_seq_len])

      def setup_texts(self, texts):
        self.texts = []
        for text in texts:
          self.texts.append(self.to_onehot(text))

      def __len__(self):
        return len(self.strokes)

      def __getitem__(self, idx):
        t = torch.as_tensor(self.strokes[idx])
        return t[:-1], self.texts[idx], t[1:]


    import string

    class HWModel(pl.LightningModule):
      def __init__(self, lr=args.lr, bs = args.batch_size,
            num_components=args.num_components,
            disable_conditional = args.disable_conditional,
            K = args.K):
        super().__init__()
        self.hidden_size = args.rnn_hidden_size
        self.K = K
        print(f"disable_conditional: {disable_conditional}")
        self.disable_conditional = disable_conditional
        if disable_conditional:
            self.rnn1 = nn.GRU(input_size=3, hidden_size=self.hidden_size, batch_first=True)
        else:
            self.rnn1 = nn.GRU(input_size=3+len(CHARS)+1, hidden_size=self.hidden_size, batch_first=True)
        self.weight_fc = nn.Linear(self.hidden_size, K*3) 
        self.fc = nn.Linear(self.hidden_size, 1+6*num_components+3*K) 
        self.num_components = num_components

        self.learning_rate = lr
        self.bceWithLogitsLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.bs = bs

        # Return the hidden tensor(s) to pass to forward
      def getNewHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

      @staticmethod
      def window_weight(window_params, u, verbose=False):
        # window_params is of shape: bs X K X 3
        alphas = window_params[:, :, 0]
        betas = window_params[:, :, 1]
        kappas = window_params[:, :, 2]
        #print(f'window_params: {window_params}')
        #print(f'u: {u}')
        #print(f'(kappas - u), {kappas - u}')
        #print(f'-(kappas - u)**2 * betas, {-(kappas - u)**2 * betas}')
        #print(f'torch.exp(-(kappas - u)**2 * betas), {torch.exp(-(kappas - u)**2 * betas)}')
        #print(f'alphas * torch.exp(-(kappas - u)**2 * betas), {alphas * torch.exp(-(kappas - u)**2 * betas)}')
        result = torch.sum(alphas * torch.exp(-(kappas - u)**2 * betas), dim=-1)
        if verbose:
            print(f'window_weight: result.shape: {result.shape}')
        return result

      @staticmethod
      # Calculates window for a single timestep
      def window(window_params, one_hot_text, verbose=False):
        # window_params shape: bs X K X 3
        # one_hot_text: shape: bs X text_len X num_chars
        if verbose:
            #print(f'window: params.shape: {window_params.shape} one_hot_text.shape: {one_hot_text.shape}')
            print(f'window: params: {window_params} ')
            #print(f'window: one_hot_text: {one_hot_text} ')
        if window_params.isnan().any():
            print(f'window(); window_params has NaN: {window_params}')
            1//0
        if one_hot_text.isnan().any():
            print(f'window(); one_hot_text has NaN: {one_hot_text}')
            1//0
        # alphas, betas, kappas: shape: bs X K
        bs, text_len, num_chars = one_hot_text.shape
        vector = torch.zeros(bs, num_chars).type_as(one_hot_text)
        for i in range(text_len):
            weight = HWModel.window_weight(window_params, i, verbose=verbose)
            #print(f'weight(t={i}): {weight}')
            vector = vector + (
                weight.resize(bs, 1) * one_hot_text[:, i])
            if verbose:
                print(f"after {i}: vector = {vector}")
            if vector.isnan().any():
                print(f'window loop({i}); vector has NaN: {vector}')
                print(f'window: params.shape: {window_params.shape} one_hot_text.shape: {one_hot_text.shape}')
                print(f'window: params: {window_params} ')
                print(f'window: one_hot_text[:, i]: {one_hot_text[:, i]} ')
                print(f'window weight: {HWModel.window_weight(window_params, i, verbose=False)}')
                1//0
        if verbose:
            print(f'window() returning: {vector}')
        if vector.isnan().any():
            print(f'window(); returning has NaN: {vector}')
            print(f'window: params.shape: {window_params.shape} one_hot_text.shape: {one_hot_text.shape}')
            print(f'window: params: {window_params} ')
            print(f'window: one_hot_text: {one_hot_text} ')
            1//0
        return vector

      def forward(self, data, hidden, verbose=False):
        if verbose:
            print(f'Forward: data[0].shape: {data[0].shape}')
        (x, w, current_kappa, text) = data
        bs, seq_len, _ = x.shape
        if current_kappa is None:
            current_kappa = torch.zeros((bs, self.K), device=self.device)
        else:
            current_kappa = current_kappa.clone()
        if w is None:
            w = torch.zeros((bs, seq_len+1, len(CHARS)+1), device=self.device)
        elif len(w.shape) == 2:  # pass in single w
            w = w.unsqueeze(dim=1)

        assert current_kappa.shape == (bs, self.K)
        assert len(w.shape) == 3
        assert w.shape[0] == bs
        assert w.shape[1] > 0
        assert w.shape[2] == len(CHARS)+1

        if verbose:
            print(f'Forward: data: {x.shape}, w: {w.shape}, text: {text.shape}')

        # we store the wparams which are the output (along with xs) of the forward
        wparams = torch.zeros((bs, seq_len, self.K*3), device=self.device)

        # The return result
        xs = torch.zeros((bs, seq_len, self.fc.weight.shape[1]), device=self.device)

        # The inputs to each timestep, calculated at previous timestep
        ws = torch.zeros((bs, seq_len+1, len(CHARS)+1), device=self.device)

        # Initial timestep: w should be zero at beginning of sequence
        ws[:, 0:1, :] = w[:, -1: :]
        for i in range(seq_len):
            if verbose:
                print(f'Forward: loop {i}')

            # extract x for current timestep
            xsmall = x[:, i:i+1, :]
            # extract window for current timestep
            w = ws[:, i:i+1, :]
            #print(f'loop: xsmall.shape {xsmall.shape} w.shape: {w.shape}')
            if self.disable_conditional:
                (xsmall, hidden) = self.rnn1(xsmall, hidden)
            else:
                cat = torch.cat((xsmall, w), dim=-1)
                #print(f'cat: {cat.shape}')
                (xsmall, hidden) = self.rnn1(cat, hidden)
            if xsmall.isnan().any():
                print(f'0; x has NaN: {xsmall}')
                1//0
            #print(f'xsmall.shape: {xsmall.shape}')
            
            # Calculate weight parameters from rnn output 
            flat_wparam = self.weight_fc(xsmall)
            #print(f'flat_wparam: {flat_wparam}')
            flat_wparam = torch.exp(flat_wparam)                # convert to non-negative alpha, beta, kappa
            #print(f'flat_wparam.shape: {flat_wparam.shape}')
           
            # Save for this timestep
            wparams[:, i:i+1, :] = flat_wparam
            wparam = flat_wparam.resize(bs, self.K, 3)

            # kappa values are relative; convert to absolute
            current_kappa += wparam[:, :, 2]
            wparam_with_total = wparam.clone()
            wparam_with_total[:, :, 2] = current_kappa

            # Save output window
            ws[:, i+1:i+2, :] = self.window(wparam_with_total, text, verbose=False).unsqueeze(dim=1)
            if ws.isnan().any():
                print(f' ws has NaN: {ws}')
                1 // 0
            if xsmall.isnan().any():
                print(f' xsmall has NaN: {xsmall}')
                1 // 0
            xs[:, i:i+1, :] = xsmall

        x = self.fc(xs)
        if x.isnan().any():
            print(f'3; x has NaN: {x}')
            1//0
        nc = self.num_components
        # Order of output is:
        # 0: penup/down
        # [1:1+num_components]: mean-x
        # [1+num_componennts:1+2*num_components]: mean-y
        # [1+2*num_components: 1+3_num_components: std-x
        # [1+3_num_components: 1+4_num_components: std-y
        # [1+3_num_components: 1+4_num_components: correlation
        # [1+4_num_components: 1+5_num_components: weighting factor
        # [1=5_num_components: 1+6_num_components + 3*K]: K values
        x_new = x.clone()

        x_new[:,:, 1+2*nc:1+4*nc] = torch.exp(x[:,:, 1+2*nc:1+4*nc])   # convert to non-negative std dev
        x_new[:,:, 1+4*nc:1+5*nc] = F.softmax(x[:,:, 1+4*nc:1+5*nc], dim=2)   # convert to %
        x_new[:,:, 1+5*nc:1+6*nc] = torch.tanh(x[:,:, 1+5*nc:1+6*nc])   # convert to range (-1, 1)
        x_new[:,:, 1+5*nc:1+6*nc] = torch.tanh(x[:,:, 1+5*nc:1+6*nc])   # convert to range (-1, 1)
        x_new[:,:, 1+6*nc:] = wparams
        if x_new.isnan().any():
            print(f'x_new has NaN: {x_new}')
            1//0
        return x_new, ws, current_kappa, hidden
        
      def construct_distribution(self, yhat, verbose=False):
        nc = self.num_components
        bs, seq_len, _ = yhat.shape
        # takes all but the penup/down part of the yhat
        mean = yhat[:, :, 0*nc:2*nc]
        mean=torch.reshape(mean, (bs, seq_len, nc, 2))
        stddev = yhat[:, :, 2*nc:4*nc]
        stddev=torch.reshape(stddev, (bs, seq_len, nc, 2))
        weights = yhat[:, :, 4*nc:5*nc]
        correlation = yhat[:, :, 5*nc:6*nc]
        covariance = torch.ones(bs, seq_len, nc, 2, 2, device=self.device)
        if verbose:
            print(f'construct_distrubution: mean: {mean}')
            print(f'construct_distrubution: stddev: {stddev}')
            print(f'construct_distrubution: weights: {weights}')
            print(f'construct_distrubution: correlation: {correlation}')
        minor_diagonal = stddev[:, :, :, 0] * stddev[:, :, :, 1] * correlation
        #print(f'minor_diagonal shape: {minor_diagonal.shape}')
        # Set major diagonal
        covariance[:, :, :, 0, 0] = stddev[:, :, :, 0] **2
        covariance[:, :, :, 1, 1] = stddev[:, :, :, 1] **2
         # Set minor diagonal
        covariance[:, :, :, 1, 0] = 0 #minor_diagonal
        covariance[:, :, :, 0, 1] = 0 #minor_diagonal
        #print(f'construct_distribution: covariance: {covariance}')
        #print(f'construct_distribution: Categorical weights: {weights}')
        mix = D.Categorical(weights, validate_args=True)
        comp = MultivariateNormal(mean, covariance, validate_args=True)
        gmm = D.mixture_same_family.MixtureSameFamily(mix, comp, validate_args=True)
        #print(f'covariance shape: {covariance.shape}')
        return gmm

      def xy_loss(self, yhat, y, verbose=False):
        #print(f'xy_loss')
        if yhat.isnan().any():
            print(f'xy_loss; yhat has NaN: {yhat}')
        #print(f'xy_loss: yhat: {yhat}')
        mc = self.construct_distribution(yhat, verbose=False)
        losses = mc.log_prob(y)
        #print(f'loss shape: {losses.shape}')
        return -losses

      def loss(self, y_hat, y):
        #print(f'y_hat.shape, y.shape: {y_hat.shape} {y.shape}')
        #print(f'y_hat: [{y_hat[:,:20:]}, {y[:,:20,:]}')
        xy_loss = self.xy_loss(y_hat[:,:,1:], y[:,:,1:]) 
        bce_loss = self.bceWithLogitsLoss(y_hat[:,:,0], y[:,:,0])
        #print(f'xy_loss: {xy_loss.shape}, bce_loss = {bce_loss.shape}')
        result = xy_loss + args.weight_updown_loss*bce_loss
        #print(f'xy_loss: result: {result.shape}')
        return result

      def training_step(self, batch, batch_idx, hiddens=None, verbose=False):
        if False:
            print(f'training_step: batch: {batch}')
            print(f'training_step: batch[0]: {batch[0]}')
            print(f'training_step: batch[1]: {batch[1]}')
            print(f'training_step: batch[2]: {batch[2]}')
            print(f'training_step: len(batch): {len(batch)}')
            print(f'training_step: type(batch): {type(batch)}')
        (data, text, y) = batch

        # mask out values where the penup < 0 (padded values are -1)
        # mask will be of shape (bs, seq_len)
        mask = torch.where(y[:,:,0] < 0, 0.0, 1.0)
        
        bs, seq_len, _ = data.shape
        #print(f'batch: {bs}, {seq_len}')
        seq_start = 0
        tbptt = args.tbptt
        hidden = self.getNewHidden(batch_size=data.shape[0])
        sub_batches = 0
        losses = []
        ws = None
        current_kappa = None
        print(f'\ntraining_step: first text: {HWPairsDataset.from_onehot(text[0])}')
        while seq_start < seq_len:
            tbptt_to_use = tbptt + random.choice([0, 1, 2])
            sub_batch = (data[:, seq_start:seq_start + tbptt_to_use, :],
                         y[:, seq_start:seq_start + tbptt_to_use, :])
            #print(f'sub_batch: {(sub_batch[0].shape, sub_batch[1].shape)}')
            
            #print(f"data.shape {data.shape}")
            #print(f"y.shape {y.shape}")
            #print(f'training_step(data.shape={data.shape}, batch_idx={batch_idx}')
            y_hat, ws, current_kappa, hidden = self.forward((sub_batch[0], ws, current_kappa, text), hidden, verbose=False)
            if False and seq_start == 0:
                print(f"training_step: seq_start: {seq_start}..{seq_start + tbptt_to_use}  ")
                print(f"kappas (summed): {torch.sum(y_hat[0,:,-1])}")
                top_vals, top_indices = torch.topk(ws, k=2)
                print(f"ws (top k=2): {top_vals[0,1:3,:], top_indices[0,1:3,:]}")
                print(f"current_kappa: {current_kappa[0]}")
            #print(f'y_hat.shape: {y_hat.shape}, sub_batch[1].shape: {sub_batch[1].shape}')
            hidden.detach()
            current_kappa.detach()
            losses.append(self.loss(y_hat, sub_batch[1]))
            #print(f'loss: {loss}')
            seq_start += tbptt_to_use
            sub_batches += 1
        unmasked_losses = torch.cat(losses, dim=1)
        masked_losses = unmasked_losses * mask
        avg_loss = torch.sum(masked_losses) / torch.sum(mask)
        self.log('loss_train', avg_loss, prog_bar=True)
        return avg_loss

      def validation_step(self, batch, batch_idx):
        #print(f'validation_step (batch={batch}, batch_idx={batch_idx}')
        data, text, y = batch
        bs, seq_len, _ = data.shape
        w = torch.zeros((bs, len(CHARS)+1), device=self.device)
        hidden = self.getNewHidden(batch_size=data.shape[0])
        current_kappa = None
        #print(f'validation_step: data.shape: {data.shape}, w.shape: {w.shape}')
        y_hat, w, current_kappa, hidden = self.forward((data, w, current_kappa, text), hidden)
        loss = torch.mean(self.loss(y_hat, y))
        #print(f'validation-step: called forward')
        self.log("loss_val", loss)
        #self.logger.experiment.add_scalars("losses", {"val_loss": losses["total"]})
        #self.log('train_loss', losses, prog_bar=True)
        #self.log("val_accuracy", 100. * c / t)
        if batch_idx == 0 and (self.current_epoch % args.sample_every_k_epochs) == 0:
          prompt = "Hello, world"
          sample, wparams = self.generate_conditionally(prompt=prompt)
          #print(f'sample: {sample}')
          print(f'generated HW epoch: {self.current_epoch}')
          f = plot_stroke(sample, prompt=None, remainder=None)
          self.logger.experiment.add_figure('generated HW', f, self.current_epoch)

          u = torch.tensor([[u] for u in range(len(prompt))], device=self.device)
          T = sample.shape[0]
          heatmap=torch.zeros((len(prompt), T))
          #print(f'heatmap shape: {heatmap.shape}')
          for t in range(T):
              window_params = wparams[t].repeat(len(prompt), 1).resize(len(prompt), self.K, 3)
              heatmap[:, t] = HWModel.window_weight(window_params, u);
          f = plot_heatmap(heatmap)
          self.logger.experiment.add_figure('window weights', f, self.current_epoch)

        return loss

      def sample_from_prediction(self, prediction):
        nc = self.num_components
        result = torch.zeros((3))
        mc = self.construct_distribution(prediction[:,:,1:])
        result[1:] = mc.sample()       
        prediction = prediction.cpu()
        result[0] = 1 if  torch.sigmoid(prediction[0, 0, 0]) > random.random() else 0
        return result

      def generate_conditionally(self, prompt: str, output_length=100, verbose=False):
        hidden = self.getNewHidden(batch_size=1)
        output = torch.zeros((output_length+1, 3), device=self.device)

        w = None
        wparams = torch.zeros(output_length, 3*self.K, device=self.device)
        prompt_tensor = HWPairsDataset.to_onehot(prompt)
        prompt_tensor = prompt_tensor.unsqueeze(dim=0)

        #print(f'generate_conditionally: prompt_tensor.shape: {prompt_tensor.shape}')
        prompt_tensor = prompt_tensor.type_as(hidden[0])
        current_kappa = torch.zeros((1, self.K), device=self.device)

        for idx in range(output_length):
          with torch.no_grad():
            input = torch.reshape(output[idx], (1, 1, 3))
            predictions, w, new_current_kappa, hidden = self.forward((input, w, current_kappa, prompt_tensor), hidden, verbose=False)
          # Only use the last prediction.
          output[idx+1] = self.sample_from_prediction(predictions[:, -1:, :])
          wparams[idx, :] = predictions[:, -1, -3*self.K:]
          wparams[idx, 2::3] += current_kappa[0, :]
          current_kappa = new_current_kappa

        #skip first (zero) element
        output = output[1:,:]
        asNumpy = output.cpu().numpy()
        # denormalize:
        asNumpy[:,1:] = asNumpy[:,1:] * self.stats['std'] + self.stats['mean']

        return asNumpy, wparams


      def generate_unconditionally(self, prompt=None, 
                            output_length=args.generated_length,
                            verbose=False):
        hidden = self.getNewHidden(batch_size=1)
        output = torch.zeros((output_length+1, 3), device=self.device)

        if prompt is not None:
            prompt = prompt.type_as(hidden)
            with torch.no_grad():
                input = torch.unsqueeze(prompt, 0)
            if verbose:
                print(f"prompt: input to forward: {input}")
            predictions, hidden = self.forward(input, hidden)
            if verbose:
                print(f"predictions: {predictions}")
            output[0] = self.sample_from_prediction(predictions[:, -1:, :])

        for idx in range(output_length):
          with torch.no_grad():
            input = torch.reshape(output[idx], (1, 1, 3))
            predictions, hidden = self.forward(input, hidden)

          # Only use the last prediction.
          output[idx+1] = self.sample_from_prediction(predictions[:, -1:, :])


        if prompt is not None:
            #skip first (zero) element
            output = output[1:,:]
        asNumpy = output.cpu().numpy()
        # denormalize:
        asNumpy[:,1:] = asNumpy[:,1:] * self.stats['std'] + self.stats['mean']

        return asNumpy

      def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer], []

      def prepare_data(self):
          datadir = hwdir / 'data'
          strokes = np.load(datadir / 'strokes-py3.npy', allow_pickle=True)
          self.strokes = strokes
          with (datadir / 'sentences.txt').open() as f:
              texts = f.readlines()
          self.texts = [t.strip() for t in texts]

      @staticmethod
      def dataloader_collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        xs = [d[0] for d in data]
        ts = [d[1] for d in data]
        ys = [d[2] for d in data]

        # TODO(neil): instead of chopping x/y off, pad with -1s
        chop_len_t = min([t.shape[0] for t in ts])
        ts = [t[:chop_len_t] for t in ts]
        tresult = torch.stack(ts, dim=0)
        if args.masking:
            max_len = max([x.shape[0] for x in xs])
            xresult = torch.ones((len(xs), max_len, 3)) * -1
            print(f"xresult shape: {xresult.shape}")
            for i, x in enumerate(xs):
                xresult[i, :x.shape[0], :] = x
            yresult = torch.ones((len(ys), max_len, 3)) * -1
            for i, y in enumerate(ys):
                yresult[i, :y.shape[0], :] = y
        else:
            # Must have same seq_len for each item in batch
            # So, chop them off
            chop_len = min([x.shape[0] for x in xs])
            xs = [x[:chop_len] for x in xs]
            ys = [y[:chop_len] for y in ys]
            xresult = torch.stack(xs, dim=0)
            yresult = torch.stack(ys, dim=0)
                
        result = (xresult, tresult, yresult)
        if xresult.isnan().any():
            print('dataloader_collate_fn: xresult nan: {xresult}')
        if tresult.isnan().any():
            print('dataloader_collate_fn: tresult nan: {tresult}')
        if yresult.isnan().any():
            print('dataloader_collate_fn: yresult nan: {yresult}')
        return result   
        #print(f'collate_fn: {xs[0].shape[0]}..{xs[-1].shape[0]}')
                  
      def train_dataloader(self):
          return DataLoader(self.train_ds, shuffle=False, batch_size=self.bs, collate_fn=self.dataloader_collate_fn, num_workers=1)

      def val_dataloader(self):
          return DataLoader(self.val_ds, shuffle=False, batch_size=self.bs, collate_fn=self.dataloader_collate_fn, num_workers=1)

      def setup(self, stage = None):
          train_split = int(len(self.strokes)*0.95)
          #valid_split = len(self.strokes) - train_split
          #train_strokes, valid_strokes = random_split(self.strokes, [train_split, valid_split])
        
          train_strokes = self.strokes[:min(args.max_training_samples, train_split)]
          valid_strokes = self.strokes[train_split:train_split + args.max_validation_samples]
          train_texts = self.texts[:min(args.max_training_samples, train_split)]
          valid_texts = self.texts[train_split:train_split + args.max_validation_samples]

          def calc_stats(values):
            totals = None
            meanUps = np.concatenate([stroke[:, 0] for stroke in values]).mean()
            shapes = [stroke[:, 1:] for stroke in values]
            totals = np.concatenate(shapes)
            return {'mean': totals.mean(axis=0), 'std': totals.std(axis=0), 'meanUps':meanUps}

          self.stats = calc_stats(train_strokes)
          #print(f'stats: {self.stats}')

          self.train_ds = HWPairsDataset(train_strokes, train_texts, self.stats)
          self.val_ds = HWPairsDataset(valid_strokes, valid_texts, self.stats)
          #print(f'HWDataModule: len(train_ds) = {len(self.train_ds)}, len(val_ds)={len(self.val_ds)}')


    model = HWModel()

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="/data/neil/runs", name="hw_generation-lightning")

    lr_monitor = LearningRateMonitor(logging_interval='step')


    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=[2],
        logger=logger,
        log_every_n_steps=1,
        )

    logger.log_hyperparams(args)

    trainer.fit(model)    


def main():
    global args
    parser = argparse.ArgumentParser(description='train conditional HW generation')
    # for model
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=768)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--rnn_hidden_size', type=int, default=900)
    parser.add_argument('--tbptt', type=int, default=20)
    parser.add_argument('--weight_updown_loss', type=float, default=2.5)
    parser.add_argument('--max_seq_len', type=int, default=1000)
    parser.add_argument('--max_training_samples', type=int, default=10000)
    parser.add_argument('--max_validation_samples', type=int, default=1000)
    parser.add_argument('--num_components', type=int, default=3)
    parser.add_argument('--sample_every_k_epochs', type=int, default=2)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--masking', action='store_true')
    parser.add_argument('--no-masking', dest='masking', action='store_false')
    parser.set_defaults(masking=True)
    parser.add_argument('--disable_conditional', action='store_true')
    parser.add_argument('--no-disable_conditional', dest='disable_conditional', action='store_false')
    parser.set_defaults(disable_conditional=False)
    # for generation
    parser.add_argument('--generated_length', type=int, default=300)
    args = parser.parse_args()
    print(args)
    run(args)


# %%
if __name__ == '__main__':
    print('starting main')
    main()
