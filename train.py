import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from config import *

from dataset.total_text import TotalText
from dataset.coco_text import COCO_Text
from dataset.synth_text import SynthText

from network.loss import UHT_Loss
from network.uht_net import UHT_Net
from util.augmentation import BaseTransform, Augmentation
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from visualize import visualize_network_output

def save_model(model, epoch, lr, optimzer):

    exp_name = cfg.dataset_name + '_' + cfg.backbone
    save_dir = os.path.join(cfg.save_directory, exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.multi_gpu else model.module.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    losses = AverageMeter()
    reg_losses = AverageMeter()
    center_loss = AverageMeter()
    region_loss = AverageMeter()

    model.train()


    print('Epoch: {} : LR = {}'.format(epoch, optimizer.param_groups[0]['lr']))

    for i, (img, reg_mask, meta) in enumerate(train_loader):
        scheduler.step()
        if img is None:
            print("Exception loading data! Preparing loading next batch data!")
            continue

        img, reg_mask = to_device(img, reg_mask)

        output = model(img)
        loss_reg, loss_dice_center, loss_dice_region = criterion(output, reg_mask)
        loss = loss_reg + loss_dice_center + loss_dice_region

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        reg_losses.update(loss_reg.item())
        center_loss.update(loss_dice_center.item())
        region_loss.update(loss_dice_region.item())

        if cfg.visualization and i % cfg.visualization_frequency == 0:
            visualize_network_output(img, output, reg_mask, mode='train')

        print(
            '[{:d} | {:d}] - Loss: {:.4f} - Reg_Loss: {:.4f} - Center_Dice_Loss: {:.4f} - Region_Dice_Loss: {:.4f} - LR: {:e}'.format(
                i, len(train_loader), loss.item(), loss_reg.item(), loss_dice_center.item(),
                loss_dice_region.item(), optimizer.param_groups[0]['lr'])
        )

    if epoch % cfg.save_frequency == 0:
        save_model(model, epoch, scheduler.get_lr(), optimizer)



def validation(model, valid_loader, criterion):
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        reg_losses = AverageMeter()
        center_loss = AverageMeter()
        region_loss = AverageMeter()

        for i, (img, reg_mask, meta) in enumerate(valid_loader):
            img, reg_mask = to_device(img, reg_mask)

            output = model(img)

            loss_reg, loss_dice_center, loss_dice_region = criterion(output, reg_mask)
            loss = loss_reg + loss_dice_center + loss_dice_region

            losses.update(loss.item())
            reg_losses.update(loss_reg.item())
            center_loss.update(loss_dice_center.item())
            region_loss.update(loss_dice_region.item())

            if cfg.visualization and i % cfg.visualization_frequency == 0:
                visualize_network_output(img, output, reg_mask, mode='val')

            print(
                'Validation: - Loss: {:.4f} - Reg_Loss: {:.4f} - Center_Dice_Loss: {:.4f} - Region_Dice_Loss: {:.4f}'
                    .format(loss.item(), loss_reg.item(), loss_dice_center.item(), loss_dice_region.item())
            )

        print('Validation Loss: {}'.format(losses.avg))
        print('Regression Loss: {}'.format(reg_losses.avg))
        print('Center Dice Loss: {}'.format(center_loss.avg))
        print('Region Dice Loss: {}'.format(region_loss.avg))


def main():

    global lr
    trainset = None
    valset = None

    if cfg.dataset_name == 'total_text':

        trainset = TotalText(
            data_root=cfg.dataset_root,
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )

        valset = TotalText(
            data_root=cfg.dataset_root,
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.dataset_name == 'coco_text':
        trainset = COCO_Text(
            data_root=cfg.dataset_root,
            is_training=True,
            transform=Augmentation(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.dataset_name == 'synth_text':
        trainset = SynthText(
            data_root='/home/andrew/Documents/Dataset/SynthText/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    else:
        pass

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.training_num_workers, pin_memory=True, timeout=10)
    if valset:
        val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.training_num_workers)
    else:
        val_loader = None

    model = UHT_Net(pretrained=True)
    if cfg.multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume != "":
        load_model(model, cfg.resume)
    criterion = UHT_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * cfg.decay_epoch, gamma=cfg.decay_rate)

    print('Start training Model.')
    for epoch in range(cfg.start_epoch, cfg.end_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch)
        if valset:
            validation(model, val_loader, criterion)

    print('End.')

if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    global cfg
    cfg = init_cfg()
    main()
