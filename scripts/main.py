'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-23 22:23:46
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:54:42
 # @ Description: Train script.
 '''

import time
import sys
sys.path.append('.')  # run from project root
import argparse
import os
import pickle
import pprint

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from utils import build_model, load_dataset
from utils import build_logger, load_config, postprocess
from utils import generate_video, visualize, calc_avg_iou

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/config.yaml', help='config file path')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='train or evaluate')
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
    model = build_model(config.model)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger.info('Building Tensorboard logger.')
    tb_logger = TensorBoardLogger('checkpoints', **config.logger)

    logger.info('Building Training dataset.')
    trainset, train_loader = load_dataset(config.trainset)
    valset, val_loader = load_dataset(config.valset)

    logger.info('Building Train phase Trainer.')
    trainer = Trainer(**config.trainer, callbacks=[lr_monitor], logger=tb_logger)
    
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

def eval(config, args, logger):
    logger.info('Building Evaluation dataset.')
    dataset, data_loader = load_dataset(config.valset)

    pred_file = 'prediction.pkl'
    if args.load_pred and os.path.exists(pred_file):
        logger.info(f'Loading {pred_file} file.')
        with open(pred_file, 'rb') as f:
            prediction = pickle.load(f)
    else:
        if args.load_pred and not os.path.exists(pred_file):
            logger.info(f'Saved results {pred_file} not exists. Start to inference.')
        logger.info('Building model.')
        model = build_model(config.model)
        logger.info('Building Eval phase Trainer.')
        trainer = Trainer(**config.trainer)
        logger.info('Evaluation started')
        prediction = trainer.predict(model, dataloaders=data_loader, 
                                    ckpt_path=config.ckpt_path)

    if args.save_pred:
        with open(pred_file, 'wb') as f:
            pickle.dump(prediction, f)
        logger.info(f'Prediction results saved in {pred_file}.')

    logger.info('Post processing.')
    processed = postprocess(prediction)

    logger.info('Calculating average iou.')
    ious, avg_iou = calc_avg_iou(dataset, processed)
    logger.info('Average iou is %4f' % avg_iou)
    
    if args.save_vis:
        logger.info('Visualizing dataset.')
        frames = visualize(dataset, processed)

        out_path = os.path.join('results', '%s_%.4f_%s.mp4' % 
                        (config.exper, avg_iou, time.strftime("%Y-%m-%d-%H-%M-%S")))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        logger.info('Save visualized results in %s.' % out_path)
        generate_video(out_path, frames, ious)
    logger.info('Evaluation finished.')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    
    logger = build_logger(config, args.mode)
    logger.info(args)
    logger.info('Configuration:\n' + pprint.pformat(config))

    if args.mode == 'train':
        train(config.train, args, logger)
    elif args.mode == 'eval':
        eval(config.eval, args, logger)
