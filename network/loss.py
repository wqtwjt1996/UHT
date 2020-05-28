import torch
import torch.nn as nn
import numpy as np

class UHT_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight.view(-1) * (input - target) ** 2)

    def Dice_loss(self, gt_score, pred_score):
        inter = torch.sum(gt_score * pred_score)
        union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
        return 1. - (2 * inter / union)

    def forward(self, input, reg_mask):
        reg_pred = input[:, :]
        reg_mask = reg_mask.unsqueeze(1)

        reg_pred_line = reg_pred.contiguous().view(-1)
        reg_mask_line = reg_mask.contiguous().view(-1)

        center_mask = torch.where(reg_mask >= 0.9, torch.full_like(reg_mask, 1), torch.full_like(reg_mask, 0))
        region_mask = torch.where(reg_mask > 0.05, torch.full_like(reg_mask, 1), torch.full_like(reg_mask, 0))
        center_pred = torch.where(reg_pred >= 0.9, torch.full_like(reg_mask, 1), torch.full_like(reg_mask, 0))
        region_pred = torch.where(reg_pred > 0.05, torch.full_like(reg_mask, 1), torch.full_like(reg_mask, 0))

        pos = reg_mask.view(-1).cpu().numpy()[np.where(reg_mask.view(-1).cpu().numpy() > 0.0)].shape[0]
        neg = reg_mask.view(-1).cpu().numpy()[np.where(reg_mask.view(-1).cpu().numpy() == 0.0)].shape[0]
        total = reg_mask.view(-1).cpu().numpy().shape[0]
        weight = torch.where(reg_mask > 0.0, torch.full_like(reg_mask, neg / total), torch.full_like(reg_mask, pos / total))

        loss_reg = self.weighted_mse_loss(reg_pred_line, reg_mask_line, weight)
        loss_dice_center = self.Dice_loss(center_mask, center_pred)
        loss_dice_region = self.Dice_loss(region_mask, region_pred)

        return loss_reg, loss_dice_center, loss_dice_region