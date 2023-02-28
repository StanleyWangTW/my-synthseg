import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np

from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from utils import plot_image
from utils import transforms
from utils import losses, models

device = "cuda" if torch.cuda.is_available() else "cpu"


class MRIData(Dataset):
    def __init__(self, dir: str):
        self.paths = glob(dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.FloatTensor:
        return torch.from_numpy(nib.load(self.paths[index]).get_fdata())[None, None, ...].float()


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

dataset = MRIData(r"nifti_files\samseg.nii.gz")

label = nib.load(r"nifti_files\samseg.nii.gz").get_fdata()
label = torch.from_numpy(label).float().to(device)

learning_rate = 1e-4
model = models.Unet3D(1, len(index_all), 24).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = losses.DiceLoss()

if not os.path.exists("saves"):
    os.makedirs("saves")
bestmodel_path = r"saves\bestmodel.pth"
checkpoint_path = r"saves\checkpoint.pth.tar"

if input("Load checkpoint ? [y/n]") == "y" and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    loss_list = checkpoint['loss']
    dice_list = checkpoint['dice']
    best_score = checkpoint['best_score']
else:
    start_step = 0
    dice_list = []
    loss_list = []
    best_score = -1



print(start_step)
savestep = 3
steps = 10 * savestep
print(f"Number of Steps: {steps}, Learning Rate: {learning_rate}")
print(f"Optimizer: {optimizer}")
print(f"Loss Function: {loss_fn}")
print("Start Training...\n")
torch.cuda.empty_cache()
model.train()
idx = 0
for i in range(steps//savestep):
    for j in tqdm(range(savestep)):
        mask = label_transform(dataset[idx].to(device))
        image = image_transform(mask)
        mask = transforms.split_labels(mask, index_all)

        pred = model(image)
        loss = loss_fn(pred, mask) # soft dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        idx += 1
        idx = idx if idx < len(dataset) else 0

    with torch.no_grad():
        step = (i+1) * savestep + start_step
        pred[pred > 0.5]  = 1 # pull to the closest value
        pred[pred <= 0.5] = 0
        dice_score = losses.dice(pred, mask).item() # hard dice
        print(f"[{step}/{steps + start_step}] Dice: {dice_score}, Loss: {loss.item()}")
        if best_score == -1 or dice_score > best_score:
            best_score = dice_score
            torch.save(model, bestmodel_path)
            print("! save best model !")
        
        loss_list.append(loss.item())
        dice_list.append(dice_score)

        # save checkpoint
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_list,
            'dice': dice_list,
            'best_score': best_score
        }, checkpoint_path)
        print("=> save checkpoint")