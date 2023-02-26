import math
import unittest

import torch

from train import scribe_loss


class TestTrain(unittest.TestCase):

    def test_scribe_loss_penup_one_matches(self):
        prediction = torch.tensor([[[1e5, 0, 0]]])
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_scribe_loss_penup_zero_matches(self):
        prediction = torch.tensor([[[-1e5, 0, 0]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        torch.testing.assert_close(result, torch.tensor(0.0))

    def test_scribe_loss_penup_zero_nomatch(self):
        prediction = torch.tensor([[[1e5, 0, 0]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        torch.testing.assert_close(result, torch.tensor(1e5/2))

    def test_scribe_loss_penup_one_nomatch(self):
        prediction = torch.tensor([[[-1e5, 0, 0]]])
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        torch.testing.assert_close(result, torch.tensor(1e5/2))

    def test_scribe_loss_coord_differ(self):
        prediction = torch.tensor([[[1e5, 3, 4]]])
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        torch.testing.assert_close(result, torch.tensor(25/2))

    def test_scribe_loss_penup_equal_weighting(self):
        prediction = torch.tensor([[[0.0, 3, 4]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask, penup_weighting=0.5)
        expected = torch.tensor((-math.log(0.5)+25)/2)
        torch.testing.assert_close(result, expected)

    def test_scribe_loss_penup_nonequal_weighting(self):
        prediction = torch.tensor([[[0.0, 3, 4]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask, penup_weighting=0.75)
        expected = torch.tensor(3 * (-math.log(0.5))/ 4 + 25/4)
        torch.testing.assert_close(result, expected)

    def test_scribe_loss_without_mask(self):
        prediction = torch.tensor([[[0.0, 3, 4], [0.0, 2, 3]]])
        target = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1, 1]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        expected = torch.tensor((-2 * math.log(0.5) + 13 + 25)/4)
        torch.testing.assert_close(result, expected)

    def test_scribe_loss_with_mask(self):
        prediction = torch.tensor([[[0.0, 3, 4], [0.0, 2, 3]]])
        target = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        target_mask = torch.tensor([[1, 0]], dtype=torch.bool)

        result = scribe_loss(prediction, target, target_mask)
        expected = torch.tensor((-math.log(0.5) + 25)/2)
        torch.testing.assert_close(result, expected)
