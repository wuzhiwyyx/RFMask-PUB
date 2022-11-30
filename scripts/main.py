'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-23 22:23:46
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:54:42
 # @ Description: Train and evaluation script.
 '''

import sys
import time

sys.path.append('.')  # run from project root
import argparse
import pickle
import pprint
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import (build_logger, build_model, load_config, load_dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/config.yaml', help='config file path')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='train or test model')
    parser.add_argument('--save_vis', action='store_true', help='save_visualized_result')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_pred", action='store_true', help='save prediction results into prediction.pkl')
    group.add_argument("--load_pred", action='store_true', help='load prediction results into prediction.pkl')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train_with_best_lr", action='store_true', help='auto find best learning rate and start training')
    group.add_argument('--best_lr', action='store_true', help='auto find best learning rate')

    return parser.parse_args()

def train(config, args, logger):
    logger.info('Building model.')
    model = build_model(**config.model)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger.info('Building metric moniter.')
    ckpt_callback = ModelCheckpoint(
        monitor='val/mask_loss', save_top_k=5, 
        filename='epoch={epoch}-step={step}-mloss={val/mask_loss:.2f}',
        auto_insert_metric_name=False,
        save_last=True, verbose=True
    )
    
    logger.info('Building Tensorboard logger.')
    tb_logger = TensorBoardLogger('checkpoints', **config.logger)

    logger.info('Building Training dataset.')
    trainset, train_loader = load_dataset(**config.trainset)
    valset, val_loader = load_dataset(**config.valset)

    logger.info('Building Train phase Trainer.')
    trainer = Trainer(**config.trainer, callbacks=[lr_monitor, ckpt_callback], logger=tb_logger)
    
    if args.best_lr:
        import matplotlib.pyplot as plt
        logger.info('Seaching for best learning rate ...')
        lr_finder = trainer.tuner.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        plt.savefig('lr_curve.jpg')
        logger.info('Best learning rate found %4f' % lr_finder.suggestion())
        logger.info('Learning rate curve has been saved in lr_curve.jpg')
    elif args.train_with_best_lr:
        lr_finder = trainer.tuner.lr_find(model)
        model.learning_rate = lr_finder.suggestion()
        logger.info('Training with learning rate %4f' % model.learning_rate)
        trainer.fit(model)
    else:
        logger.info('Training started.')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=config.ckpt_path)
        logger.info('Training finished.')

def test(config, args, logger):
    logger.info('Building Evaluation dataset.')
    dataset, data_loader = load_dataset(**config.valset)

    logger.info('Building model.')
    model = build_model(**config.model, save_pred=args.save_pred, vis=args.save_vis)
    
    logger.info('Building Test phase Trainer.')
    trainer = Trainer(**config.trainer)

    logger.info('Test started.')
    trainer.test(model, dataloaders=data_loader, ckpt_path=config.ckpt_path)
    logger.info('Test finished.')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    
    logger = build_logger(config, args.mode, model_name=config.model_name)
    
    pl.seed_everything(1)
    logger.info(args)
    logger.info(f'Configuration:\n{pprint.pformat(config)}')

    if args.mode == 'train':
        train(config.train, args, logger)
    elif args.mode == 'test':
        test(config.eval, args, logger)
