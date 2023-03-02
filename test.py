import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

import argparse

from utils.plot_image import show_slices, show_labels
from utils import models, losses
from utils import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
index_all = [2,  3,   4,   5,   7,   8,   10,  11,  12,  13,  14,  15,  16,  17,
            18,  26,  28,  31,  41,  42,  43,  44,  46,  47,  49,  50,
            51, 52,  53,  54,  58,  60,  63,  77]

label_dict = {
    2:  "Left Cerebral WM",
    3:  "Left Cerebral Cortex",
    4:  "Left Lateral Ventricle",
    5:  "Left Inf Lat Vent",
    7:  "Left Cerebellum WM",
    8:  "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brain Stem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    26: "Left Accumbens area",
    28: "Left VentralDC",
    31: "Left choroid plexus",
    41: "Right Cerebral WM",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inf Lat Vent",
    46: "Right Cerebellum WM",
    47: "Right Cerebellum Cortex", 
    49: "Right Thalamus",
    50: "Right Caudate",
    51: "Right Putamen",
    52: "Right Pallidum",
    53: "Right Hippocampus",
    54: "Right Amygdala",
    58: "Right Accumbens area",
    60: "Right VentralDC",
    63: "Right choroid plexus",
    77: "WM hypointensities"
}

class MRIData(Dataset):
    def __init__(self, dir: str):
        self.paths = glob(dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.FloatTensor:
        return torch.from_numpy(nib.load(self.paths[index]).get_fdata())[None, None, ...].float()

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-I", "--input", nargs="input", type=str, help="input files")
    # parser.add_argument("--model", nargs="model_path", type=str, help="model path")
    # args = parser.parse_args()

    # dataset = MRIData(args.input)
    with torch.no_grad():

        label_transform = transforms.Compose([
            transforms.RandomSkullStrip(),
            transforms.RandomLRFlip(),
            transforms.LinearDeform(scales=(0.8, 1.2), degrees=(-20, 20), shears=(-0.015, 0.015), trans=(-30, 30)),
            transforms.NonlinearDeform(max_std=4),
            transforms.RandomCrop(120)
        ])

        image_transform = transforms.Compose([
            transforms.GMMSample(mean=(0, 255), std=(0, 35)),
            transforms.RandomBiasField(max_std=0.6),
            transforms.Rescale(),
            transforms.GammaTransform(std=0.4),
            transforms.RandomDownSample(max_slice_space=9, alpha=(0.95, 1.05), r_hr=1)
        ])

        transform = transforms.Compose([
            transforms.RandomCrop(120),
            transforms.Rescale()
        ])

        # label1 = torch.from_numpy(nib.load(r"nifti_files\label\seg.nii.gz").get_fdata()).float().to(device)[None, None, ...] 
        # label2 = torch.from_numpy(nib.load(r"nifti_files\label\samseg.nii.gz").get_fdata()).float().to(device)[None, None, ...] 
        
        # show_slices(label1[0, 0, ...], (100, 100, 100), "gist_ncar")
        # show_slices(label2[0, 0, ...], (100, 100, 100), "gist_ncar")

        # label = label_transform(label)
        # show_labels(transforms.split_labels(label, index_all), 100, 9, 5, index_all, label_dict, True, "label3.jpg")
        # gen_img = image_transform(label)

        img = torch.from_numpy(nib.load(r"nifti_files\CC0001_philips_15_55_M.nii.gz").get_fdata()).float().to(device)[None, None, ...] 
        img = transform(img)
        show_slices(img[0, 0, ...], (60, 60, 60), "gray")

        model = models.Unet3D(1, 34, 24)
        model = torch.load(r"saves\bestmodel.pth").to(device)

        pred = model(img)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        show_labels(pred, 60, 9, 5, index_all, label_dict, True, "pred2.jpg")


main()