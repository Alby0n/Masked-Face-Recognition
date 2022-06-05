import glob
import random
from typing import Tuple, Any

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
#from mask_the_face import Masker
from torch.utils.data import Dataset

class NormalizeRange(object):
    def __init__(self, minval, maxval, all_data_min=0, all_data_max=1):
        self.minval = minval
        self.maxval = maxval
        self.all_data_min = all_data_min
        self.all_data_max = all_data_max

    def __call__(self, t):
        normalized = self.minval + (self.maxval - self.minval) * (t - self.all_data_min)/(self.all_data_max - self.all_data_min)
        return normalized

class MaskedCelebADataset(Dataset):

    """
    MaskedCelebADataset contains all
    """

    def __init__(self, root_dir, image_shape, mode="train", mask_type="random", train_fraction=0.7, apply_transforms=True):
        """
        Construct MaskedCelebA dataset
        mode = (train|test)
        """
        super().__init__()

        self.root_dir = root_dir
        self.mode = mode
        self.image_shape = image_shape
        self._images = sorted(glob.glob(f"{root_dir}/images/*.??g"))
        t = int(len(self._images) * train_fraction)
        self._images = self._images[:t] if mode == "train" else self._images[t:]

        # masking attributes
        self._masker = Masker()
        self.mask_type = mask_type
        # default transforms for the image
        self.apply_transforms = apply_transforms
        self._transforms = transforms.Compose([
            #  transforms.Resize(image_shape),
            transforms.ToTensor(),
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index):
        img = cv2.imread(self._images[index])
        img = cv2.resize(img, self.image_shape)
        masked_img, mask = self._mask_image(img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masked_img = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask)

        if self.apply_transforms:
            img = self._transforms(img)
            masked_img = self._transforms(masked_img)
            #  mask = self._transforms(mask)
            mask = transforms.Compose([transforms.ToTensor(), NormalizeRange(minval=0, maxval=1)])(mask)


        #  mask = torch.Tensor([mask])
        #  mask = self.generate_mask()

        return (img, masked_img, mask)

    def _mask_image(self, img):
        """
        Apply mask to the image
        """
        masked_img, mask, _, _ = self._masker.apply_mask(img, mask_type=self.mask_type)

        return masked_img, mask


# test
if __name__ == "__main__":
    ds = MaskedCelebADataset("dataset/celeba", (256, 256), apply_transforms=True)
    print(f"{len(ds)} training images loaded ")

    img, masked_img, mask = random.choice(ds)

    print(f"img size: {img.size()}")
    print(f"masked_img size: {masked_img.size()}")
    print(f"masked_part size: {mask.size()}")

    img = transforms.ToPILImage()(img)

    img.show()
    #  masked_img.show()
    #  masked_part.show()
