import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import os
from glob import glob
import argparse

from utils.plot_image import show_slices, show_labels
from utils import models, losses
from utils import transforms

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
        self.image_paths = glob(os.path.join(dir, "images", "*"))
        if len(self.image_paths) != len(glob(os.path.join(dir, "labels", "*"))):
            raise Exception("Error: number of images and labels doesn't match")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> list:
        return [
            torch.from_numpy(nib.load(self.image_paths[index]).get_fdata())[None, None, ...].float(),
            torch.from_numpy(nib.load(self.image_paths[index].replace("images", "labels")).get_fdata())[None, None, ...].float()
        ]
    
transform = transforms.Compose([
    # transforms.RandomCrop(100),
    transforms.Rescale()
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", metavar="<model>", help="model path")
    parser.add_argument("input", metavar="<input>", help="path of input folder")
    parser.add_argument("--hard", action="store_true", help="calculate hard dice score")
    parser.add_argument("-f", "--flip", action="store_true", help="enable flip augmentation in test time")
    parser.add_argument("-g", "--use_gpu", action="store_true", help="enable gpu")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    dataset = MRIData(args.input)
    model = models.Unet3D(1, 34, 24)
    model = torch.load(args.model).to(device)

    image, label = dataset[0]

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        image = image.to(device)
        image = transform(image)
        pred = model(image)

        if args.flip:
            torch.cuda.empty_cache()
            flipped_image = image.flip(dims=(2,))
            flipped_pred = model(flipped_image)
            pred = (pred + flipped_pred.flip(dims=(2,))) / 2
        
        if args.hard:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        
        # show_labels(pred, 25, 9, 5, index_all, label_dict)
        pred = pred.argmax(dim=1)
        print(pred.unique(), len(pred.unique()), len(index_all))
        for i, l in enumerate(index_all):
            pred[pred == i+1] = l

        print(pred.unique())
        show_slices(pred[0, ...], (110, 110, 110), "gist_ncar")

        return


main()