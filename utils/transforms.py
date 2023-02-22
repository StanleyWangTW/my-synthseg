import torch
import random


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transfom in self.transforms:
            output = transfom(image)
        
        return output


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