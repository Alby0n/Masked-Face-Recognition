import multiprocessing
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import numpy as np

from network.generator import Generator
from network.patch_discriminator import PatchDiscriminator
from network.perceptual_network import PerceptualNet
from dataset import MaskedCelebADataset

IMAGE_SIZE=128

class SNPatchGAN(LightningModule):
    def __init__(self,
        latent_dim: int = 100,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        weight_decay: float = 0.0,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        discriminator_train_frequency: int = 5,
        lambda_l1: float = 100,
        lambda_perceptual: float = 10,
        lambda_gan: float = 1,
        train_dataset_path: str = "dataset/ffhq",
        val_dataset_path: str = "dataset/celeba",
        face_mask_type: str = 'random',
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.weight_decay = weight_decay
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.discriminator_train_frequency = discriminator_train_frequency
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gan = lambda_gan
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.face_mask_type = face_mask_type

        self.example_input_array = (
            torch.ones((self.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)),
            torch.ones((self.batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)),
        )

        self.generator = Generator()
        self.discriminator = PatchDiscriminator()
        self.perceptual_net = PerceptualNet()

        self.weight_init()

    def weight_init(self, init_gain: float = 0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                nn.init.xavier_normal_(m.weight.data, gain = init_gain)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('Linear') != -1:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.apply(init_func)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': self.discriminator_train_frequency},
        )

    def train_dataloader(self):
        dataset = MaskedCelebADataset(self.train_dataset_path, (IMAGE_SIZE, IMAGE_SIZE), mode="train", train_fraction=0.9, mask_type=self.face_mask_type)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count(), drop_last=True)

    def val_dataloader(self):
        dataset = MaskedCelebADataset(self.val_dataset_path, (IMAGE_SIZE, IMAGE_SIZE), mode="val", train_fraction=0, mask_type=self.face_mask_type)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count(), drop_last=True, shuffle=False)

    def forward(self, img, mask):
        return self.generator(img, mask)

    def training_step(self, batch, batch_idx, optimizer_idx):
        img, _, mask = batch

        # train generator
        if optimizer_idx == 0:
            # generate images
            coarse_img, refined_img = self(img, mask)
            completed_img = img * (1 - mask) + refined_img * mask

            # L1 loss
            coarse_l1loss = (coarse_img - img).abs().mean()
            refined_l1loss = (refined_img - img).abs().mean()
            # local loss
            fake_scalar = self.discriminator(completed_img, mask)
            gan_loss = -torch.mean(fake_scalar)
            # global loss
            img_features = self.perceptual_net(img)
            refined_features = self.perceptual_net(refined_img)
            perceptual_loss = F.l1_loss(refined_features, img_features)
            # complete loss
            loss_g = self.lambda_l1 * coarse_l1loss + \
                     self.lambda_l1 * refined_l1loss + \
                     self.lambda_perceptual * perceptual_loss + \
                     self.lambda_gan * gan_loss

            # log generated images
            grid = torchvision.utils.make_grid(completed_img[:6])
            self.logger.experiment.add_image('step_refined_images', grid, self.global_step)
            # log loss
            self.log("gen_total_loss", loss_g)

            tqdm_dict = {'loss_g': loss_g}
            output = OrderedDict({
                'loss': loss_g,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            coarse_img, refined_img = self(img, mask)
            completed_img = img * (1 - mask) + refined_img * mask

            valid = torch.Tensor(np.ones((self.batch_size, 1, IMAGE_SIZE//32, IMAGE_SIZE//32)))
            fake = torch.Tensor(np.zeros((self.batch_size, 1, IMAGE_SIZE//32, IMAGE_SIZE//32)))
            zero = torch.Tensor(np.zeros((self.batch_size, 1, IMAGE_SIZE//32, IMAGE_SIZE//32)))
            valid = valid.type_as(img)
            fake = fake.type_as(img)
            zero = zero.type_as(img)

            fake_scalar = self.discriminator(completed_img, mask)
            true_scalar = self.discriminator(img, mask)
            # loss
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
            # Overall Loss and optimize
            loss_d = 0.5 * (loss_fake + loss_true)

            self.log("disc_fake_loss", loss_fake)
            self.log("disc_true_loss", loss_true)
            self.log("disc_total_loss", loss_d)
            tqdm_dict = {'loss_d': loss_d}
            output = OrderedDict({
                'loss': loss_d,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        imgs, masked_imgs, masks = batch
        coarse_imgs, refined_imgs = self(imgs, masks)
        completed_imgs = imgs * (1 - masks) + refined_imgs * masks

        # sample and log images
        images = []
        coarse = []
        refined = []
        masked = []
        completed = []
        for i in range(imgs.size()[0]):
            img = imgs[i].cpu()
            coarse_img = coarse_imgs[i].cpu()
            refined_img = refined_imgs[i].cpu()
            masked_img = masked_imgs[i].cpu()
            completed_img = completed_imgs[i].cpu()
            # transform tensor -> img
            img = transforms.ToPILImage()(img)
            coarse_img = transforms.ToPILImage()(coarse_img)
            refined_img = transforms.ToPILImage()(refined_img)
            masked_img = transforms.ToPILImage()(masked_img)
            completed_img = transforms.ToPILImage()(completed_img)
            # append to grid lists
            images.append(img)
            coarse.append(coarse_img)
            refined.append(refined_img)
            masked.append(masked_img)
            completed.append(completed_img)
        # transform to grid
        grid = np.hstack([np.vstack(masked), np.vstack(coarse), np.vstack(refined), np.vstack(completed), np.vstack(images)])
        grid = np.transpose(grid, axes=[2,0,1])
        self.logger.experiment.add_image('validation_refined_images', grid, self.current_epoch)

        # global loss
        img_features = self.perceptual_net(imgs)
        refined_features = self.perceptual_net(refined_imgs)
        perceptual_loss = F.l1_loss(refined_features, img_features)

        self.log("validation_perceptual_loss", perceptual_loss)

        return perceptual_loss
