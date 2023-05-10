#!/usr/bin/env python

import time

import torch
from fvcore.common.checkpoint import Checkpointer
import wandb
from gaze_estimation.config import get_default_config
from gaze_estimation import (
    create_dataloader,
    create_logger, TotalLoss, create_model,
    create_optimizer, create_scheduler,
)
from gaze_estimation.utils import (
    AverageMeter, compute_angle_error, accuracy,
    create_train_output_dir,
    save_config, set_seeds, setup_cudnn
)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_preds_grds(pred_gazes, pred_emotions, grd_gazes, grd_emotions):
    mask_gazes = torch.where(grd_gazes[:, 0] != 0)
    mask_emotions = torch.where(grd_emotions[:, 0] != -1)

    good_pred_gazes = pred_gazes[mask_gazes]
    good_grd_gazes = grd_gazes[mask_gazes]

    good_pred_emotions = pred_emotions[mask_emotions]
    good_grd_emotions = grd_emotions[mask_emotions]

    return good_pred_gazes, good_grd_gazes, good_pred_emotions, good_grd_emotions


def train(epoch, model, optimizer, scheduler, loss_function, train_loader, config, logger):
    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    e_loss_meter = AverageMeter()
    g_loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (images, grd_gazes, grd_emotions) in enumerate(train_loader):

        images = images.to(device)
        grd_gazes = grd_gazes.to(device)
        grd_emotions = grd_emotions.to(device)
        optimizer.zero_grad()

        pred_gazes, pred_emotions = model(images)

        loss, e_loss, g_loss = loss_function(pred_gazes, pred_emotions, grd_gazes, grd_emotions.squeeze(1))
        loss.backward()

        optimizer.step()
        good_pred_gazes, good_grd_gazes, good_pred_emotions, good_grd_emotions = prepare_preds_grds(
            pred_gazes,
            pred_emotions,
            grd_gazes,
            grd_emotions
        )
        angle_error = compute_angle_error(good_pred_gazes, good_grd_gazes[:, 1:]).mean()
        emotion_accuracy = accuracy(good_pred_emotions, good_grd_emotions)[0]

        num = images.size(0)  # batch_size
        loss_meter.update(loss.item(), num)

        angle_error_meter.update(angle_error.item(), num=good_grd_gazes.size(0))
        g_loss_meter.update(g_loss.item(), num=good_grd_gazes.size(0))

        accuracy_meter.update(emotion_accuracy.item(), num=good_grd_emotions.size(0))
        e_loss_meter.update(e_loss.item(), num=good_grd_emotions.size(0))

        if step % config.train.log_period == 0:
            logger.info(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'g_loss: ({g_loss_meter.avg:.4f}) '
                        f'e_loss: ({e_loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} ({angle_error_meter.avg:.2f}) '
                        f'emo accuracy {accuracy_meter.val:.2f} ({accuracy_meter.avg:.2f}) ')
        wandb.log({
            'train/total_loss': loss_meter.avg,
            'train/e_loss': e_loss_meter.avg,
            'train/g_loss': g_loss_meter.avg,
        })

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    wandb.log({
        'train/total_loss_epoch': loss_meter.avg,
        'train/e_loss_epoch': e_loss_meter.avg,
        'train/g_loss_epoch': g_loss_meter.avg,
        'train/angle_error_train': angle_error_meter.avg,
        'train/emo_accuracy_train': accuracy_meter.avg,
    })


def validate(epoch, model, loss_function, val_loader, config, logger):
    logger.info(f'Val {epoch}')

    model.eval()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, grd_gazes, grd_emotions) in enumerate(val_loader):

            images = images.to(device)
            grd_gazes = grd_gazes.to(device)
            grd_emotions = grd_emotions.to(device)

            pred_gazes, pred_emotions = model(images)

            loss, e_loss, g_loss = loss_function(pred_gazes, pred_emotions, grd_gazes, grd_emotions.squeeze(1))

            good_pred_gazes, good_grd_gazes, good_pred_emotions, good_grd_emotions = prepare_preds_grds(
                pred_gazes,
                pred_emotions,
                grd_gazes,
                grd_emotions
            )
            angle_error = compute_angle_error(pred_gazes, grd_gazes[:, 1:]).mean()
            emotion_accuracy = accuracy(pred_emotions, grd_emotions)[0]

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num=good_grd_gazes.size(0))
            accuracy_meter.update(emotion_accuracy.item(), num=good_grd_emotions.size(0))

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f} '
                f'emo accuracy {accuracy_meter.avg:.2f} ')
    wandb.log({
        'val/total_loss_epoch_val': loss_meter.avg,
        'val/angle_error_val': angle_error_meter.avg,
        'val/emo_accuracy_val': accuracy_meter.avg,
    })

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


def main():

    path_config = r'D:\emotion_in_gaze\configs\efficientnet_train.yaml'
    config = get_default_config()
    config.merge_from_file(path_config)
    if config.train.wandb:
        wandb.init(project='Emotion_in_gaze', entity='2neurons')
    else:
        wandb.init(mode="disabled")
    wandb.run.name = config.train.output_dir.split('\\')[-1]
    wandb.config.update(config)

    set_seeds(config.train.seed)
    setup_cudnn(config)

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    # TODO: create dataset for rafdb (o sa fie mostly copy paste ca am facut asta deja) - done
    # TODO: scoate pose din datasetul de eth-xgaze ca nu am nevoie de asa ceva - done
    # TODO: concat dataset cu clasele de person - done
    # TODO: facut task indicator pentru loss in functie de targets (daca avem gaze/emo) - done
    # TODO: adus metoda de evaluare si pentru emotii, din fericire avem tot average meter deci o sa fie usor - done
    # TODO: afiseaza lossurile individual !!! - done
    # TODO: foloseste logger sa afisezi numarul de samples pt emotions vs gaze
    # TODO: afiseaza numarul de imagini per emotie
    # TODO: weight sampler daca va fi nevoie, sa speram ca nu - done
    train_loader, val_loader = create_dataloader(config)
    model = create_model(config)
    loss_function = TotalLoss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    checkpointer = Checkpointer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=output_dir.as_posix(),
        save_to_disk=True,
    )

    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config, logger)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader, config, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config, logger)

        if epoch % config.train.checkpoint_period == 0 or epoch == config.scheduler.epochs:
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)


if __name__ == '__main__':
    main()
