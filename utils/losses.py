import torch

class FocalLoss():
    def __init__(self, alpha=1, gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, pred, label):
        pt = label * pred + (1 - label) * (1 - pred)
        return -(self.alpha * torch.pow((1 - pt), self.gamma) * torch.clamp(torch.log(pt), min=-100)).mean()
    
    def __repr__(self):
        return f"Focal Loss(alpha: {self.alpha}, gamma: {self.gamma})"


def dice(pred, target, smooth=1e-5):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class DiceLoss():
    def __init__(self, smooth=1e-5):
        self.smooth = smooth
        
    def __call__(self, pred, label):
        return 1 - dice(pred, label, smooth=self.smooth)
    
    def __repr__(self):
        return f"Dice Loss(smooth: {self.smooth})"
    
class TverskyLoss():
    def __init__(self, beta=0.5, smooth=1e-5):
        self.beta = beta
        self.smooth = smooth
    
    def __call__(self, pred, label):
        label = label.flatten()
        pred = pred.flatten()
        true_pos = (label * pred).sum()
        false_pos = ((1 - label) * pred).sum()
        false_neg = (label * (1 - pred)).sum()
        return 1 - (true_pos + self.smooth) / (true_pos + self.beta * false_pos + (1 - self.beta) * false_neg + self.smooth)
    
    def __repr__(self):
        return f"Tversky Loss(beta: {self.beta}, smooth: {self.smooth})"


class FocalTversky():
    def __init__(self, beta=0.5, gamma=0.75, smooth=1e-5):
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def __call__(self, pred, label):
        return torch.pow(TverskyLoss(beta=self.beta, smooth=self.smooth)(pred, label), self.gamma)
    
    def __repr__(self):
        return f"Focal Tversky Loss(beta: {self.beta}, gamma: {self.gamma}, smooth: {self.smooth})"