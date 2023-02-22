import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_image import show_slices
from utils import transforms

label = nib.load("samseg.nii.gz").get_fdata()
label = torch.from_numpy(label).float()

show_slices(label, (100, 100, 120), "gist_ncar")

transform = transforms.Compose([
                transforms.RandomSkullStrip(),
                transforms.RandomLRFlip(),
                transforms.RandomCrop(160)
            ])

gen_label = transform(label[None, None, ...])[0, 0, ...]

show_slices(gen_label, (100, 100, 120), "gist_ncar")