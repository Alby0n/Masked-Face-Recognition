import os

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from network import SNPatchGAN

def main(args: Namespace) -> None:
    model = SNPatchGAN(**vars(args))

    #  if args.load_checkpoint is not None:
    #      model.load_from_checkpoint(args.load_checkpoint)

    # set up trainer
    trainer = Trainer(
        gpus=args.gpus,
        resume_from_checkpoint=args.resume_checkpoint,
        auto_scale_batch_size=args.auto_scale_batch_size,
        max_epochs=args.max_epochs,
        #  limit_train_batches=10,
        limit_val_batches=int(6000 / args.batch_size),
    )

    # tune params
    trainer.tune(model)

    trainer.logger.experiment.add_graph(model, model.example_input_array)

    # train
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--max_epochs", type=int, default=50, help="maximum number of epochs to train for")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--auto_scale_batch_size", type=str, choices=["power", "binsearch"], default=None, help="batch size autoscale type")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate of generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="adam: learning rate of discriminator")
    parser.add_argument("--weight_decay", type = float, default = 0, help = "Adam: weight decay")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--lambda_l1", type = float, default = 100, help = "the parameter of L1Loss")
    parser.add_argument("--lambda_perceptual", type = float, default = 10, help = "the parameter of FML1Loss (perceptual loss)")
    parser.add_argument("--lambda_gan", type = float, default = 1, help = "the parameter of valid loss of AdaReconL1Loss; 0 is recommended")

    parser.add_argument("--face_mask_type", type=str, default="surgical_blue")


    hparams = parser.parse_args()

    main(hparams)
