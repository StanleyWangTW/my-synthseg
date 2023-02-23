import torch
import torch.nn.functional as F
import random


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        output = image.clone()
        for transfom in self.transforms:
            output = transfom(output)
        
        return output

# Training Augmentations
class RandomSkullStrip():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, label):
        skull_indexs = [30, 62, 85, 165, 258, 259]
    
        output = torch.clone(label)
        if random.choice((0, 1)):
            if self.verbose: print("skull strip")
            for idx in skull_indexs:
                output[output == idx] = 0

            if random.choice((0, 1)):
                if self.verbose: print("remove CSF")
                output[output == 24] = 0
                
        return output


class RandomLRFlip():
    def __init__(self, p=0.5, verbose=False):
        self.p = p
        self.verbose = verbose
    
    def __call__(self, image):
        """input/output 5D tensor (N, C, D, H, W)"""
        if random.random() > self.p:
            if self.verbose: print("flip image")
            return image.flip(dims=(2,))
        else:
            return image


class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        """input/output tensor num_dims >= 3"""
        size = (self.size,) * 3 if type(self.size) == int else self.size

        i = random.randint(0, image.shape[-3] - size[0])
        j = random.randint(0, image.shape[-2] - size[1])
        k = random.randint(0, image.shape[-1] - size[2])
        
        return image[..., i:i+size[0], j:j+size[1], k:k+size[2]]


# Generative Model
import math
from math import sin, cos

class LinearDeform():
    """input/output pytorch 5D tensor (N, C, D, H, W)"""
    def __init__(self, scale=(0.8, 1.2), angle=(-20, 20), shear=(-0.015, 0.015), trans=(-0.3, 0.3), verbose=False):
        self.scale = scale
        self.angle = angle
        self.shear = shear
        self.trans = trans
        self.verbose = verbose
    
    def __call__(self, label):
        pd = ( max(label.shape) - label.shape[-3] )
        ph = ( max(label.shape) - label.shape[-2] )
        pw = ( max(label.shape) - label.shape[-1] )
        label = F.pad(label, (pw//2, pw - pw//2, ph//2, ph - ph//2, pd//2, pd - pd//2))
        
        scale_m = torch.diag( 1/torch.FloatTensor(3).uniform_(self.scale[0], self.scale[1]) ) # scale

        angleX = math.radians(random.uniform(self.angle[0], self.angle[1]))
        angleY = math.radians(random.uniform(self.angle[0], self.angle[1]))
        angleZ = math.radians(random.uniform(self.angle[0], self.angle[1]))
        
        
        from math import sin, cos
        rotX = torch.tensor([[cos(angleX), 0,  sin(angleX)],
                            [0.,           1.,           0.],
                            [-sin(angleX), 0.,  cos(angleX)]])

        rotY = torch.tensor([[cos(angleY), -sin(angleY), 0.],
                            [sin(angleY),  cos(angleY), 0.],
                            [0.,          0.,           1.],])

        rotZ = torch.tensor([[1.,          0.,           0.],
                            [0., cos(angleZ), -sin(angleZ)],
                            [0., sin(angleZ),  cos(angleZ)]])

        shX = random.uniform(self.shear[0], self.shear[1])
        shY = random.uniform(self.shear[0], self.shear[1])
        shZ = random.uniform(self.shear[0], self.shear[1])

        shearX = torch.tensor([[ 1.,  0.,   0.],
                            [shX,  1.,   0.],
                            [shX,  0.,   1.]])

        shearY = torch.tensor([[1.,  shY,   0.],
                            [0.,   1.,   0.],
                            [0.,  shY,   1.]])

        shearZ = torch.tensor([[1.,  0.,  shZ],
                            [0.,  1.,  shZ],
                            [0.,  0.,   1.]])

        trans = -2 * torch.FloatTensor(3).uniform_(self.trans[0], self.trans[1]) / 256

        from torch.linalg import multi_dot
        affine = torch.zeros((3, 4))
        affine[:3, :3] = multi_dot((scale_m, rotX, rotY, rotZ, shearX,shearY, shearZ))
        affine[:, 3] = trans
        
        grid = F.affine_grid(affine[None, ...], label.size(), align_corners=False).to(label.device)
        new_label = F.grid_sample(label, grid, mode='nearest', align_corners=False)

        if self.verbose:
            print(scale_m)
            print(trans)
            print(rotX)
            print(rotY)
            print(rotZ)
            print(shearX)
            print(shearY)
            print(shearZ)
            print(affine, affine.dtype)
        
        return new_label[..., pd//2:-(pd - pd//2), ph//2:256-(ph - ph//2), pw//2:256-(pw - pw//2)]


class GMMSample():
    def __init__(self, mean=(0, 255), std=(0, 35)):
        self.mean = mean
        self.std = std
    
    def __call__(self, label):
        index_all = torch.unique(label)
        gen_ima = label.clone()
        for ii in index_all:    
            mu = random.uniform(self.mean[0], self.mean[1])
            sigma = random.uniform(self.std[0], self.std[1]) # mean and standard deviation
            len1 = torch.sum(label==ii).item()
            gen_ima[label==ii] = torch.normal(mu , sigma, (len1,), device=label.device)
        
        return gen_ima


class RandomBiasField():
    def __init__(self, max_std=0.6, mode='trilinear'):
        self.max_std = max_std
        self.mode = mode
    
    def __call__(self, image):
        std = random.uniform(0, self.max_std)
        B = torch.normal(0, std, (4, 4, 4), device=image.device)
        B = F.interpolate(input=B[None, None, ...], size=image.shape[-3:], mode=self.mode)
        B = B[0, 0, ...]
        B = torch.exp(B)
        return image * B


class Rescale():
    def __init__(self, min_max=(0, 1)):
        self.min_max = min_max

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class GammaTransform():
    def __init__(self, std=0.632):
        self.std = std

    def __call__(self, image):
        return torch.pow( image, torch.exp(torch.normal(0, self.std, image.shape, device=image.device)) )


def get_gauss(sigma, kernel_size = 3):
    # Calculate the center of the kernel
    center = (kernel_size - 1) / 2

    # Create a 3D coordinate grid
    x = torch.linspace(-center, center, kernel_size)
    y = torch.linspace(-center, center, kernel_size)
    z = torch.linspace(-center, center, kernel_size)
    xx, yy, zz = torch.meshgrid(x, y, z)

    # Calculate the Gaussian function
    gaussian = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

    # Normalize the kernel so that it sums to 1
    gaussian = gaussian / torch.sum(gaussian)
    
    return gaussian

class RandomDownsample():
    def __init__(self):
        pass
        
    def __call__(img):
        r_spac = random.uniform(1, 9)
        r_thick = random.uniform(1, r_spac)
        a = random.uniform(0.95, 1.05)
        std_thick = (2 * a * math.log(10) * r_thick) / (2 * math.pi * 1)
        
        origin_shape = img.size()
        sample_shape = (torch.tensor(img.size()) / r_spac).int()
        sample_shape = torch.Size(sample_shape.tolist())
        
        gauss_kernel = get_gauss(sigma=std_thick).to(img.device)
        img = F.conv3d(img[None, None, ...], weight=gauss_kernel[None, None, ...], padding=1)
        
        # downsample to low resolution r_spac
        img = F.interpolate(input=img,  size=sample_shape, mode='trilinear')
        # upsample back to r_hr
        img = F.interpolate(input=img, size=origin_shape, mode='trilinear')
        img = img[0, 0, ...]
        
        return img