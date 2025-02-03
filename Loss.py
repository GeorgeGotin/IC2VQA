import torch
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

from Models import get_model

def get_layer(model, layer_name):
        module = model
        for name in layer_name.split('.'):
            if not name:
                break
            module = getattr(module, name)
        return module

class Loss:
    def __init__(self, device, *args, **kwargs):
        device = device[0] if isinstance(device, list) else device
        self.device = device
        self._zero_run = False

    def zero_run(self, flag : bool = True):
        self._zero_run = flag
    
    def eval(self):
        self._zero_run = False

    def __call__(self, inputs, *args, **kwargs):
        return torch.tensor(1, requires_grad=True)
    
    def to(self, device):
        self.device = device

class RegressionLoss(Loss):
    def __init__(self, model_preprocces, device='cuda', targeted=True, target=1, anchored=False, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.targeted = targeted
        device = device[0] if isinstance(device, list) else device
        self.device = device
        self.model, self.preproccess = model_preprocces
        self.target = self.model.range * target
        self.anchored = anchored
        self.register_hook()
    
    def zero_run(self, flag):
        super().zero_run(flag)
        if not self.anchored and flag:
            self.start_value = 0
            self._zero_run = False
        
    def register_hook(self):
        def forward_hook(module, input, output):
            if self._zero_run:
                self.start_value = output.clone().detach().requires_grad(False)
                self.zero_run(False)
            else:
                self.value = output.clone()

    def __call__(self):
        value = self.value

        c1 = int(self.targeted)
        c2 = int(self.anchored)
        loss = c1 * abs(value - self.target) - c2 * abs(value - self.start_value)

        return loss.mean()

    def to(self, device):
        super().to(device)
        self.start_value.to(device)
        self.target.to(device)

class CrossLayerLoss(Loss):
    def __init__(self, model_layer, device='cuda', targeted=False, target=None, anchored=True, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.anchored = anchored
        self.layer = model_layer
        self.register_hook()
        self.targeted = targeted
        if targeted:
            self.target = target.detach().clone().to(device)

    def register_hook(self):
        self.activations = None
        def forward_hook(module, input, output):
            b = input.shape[0]
            activations = output.clone()
            if isinstance(activations, list):
                dirs = []
                for activation in activations:
                    dirs.append(activation.reshape(b, -1))
                activations = torch.concat(dirs, 1)
            else:
                activations = activations.reshape(b, -1)

            if self._zero_run:
                self.zero_activations = activations.detach().requires_grad_(False)
                self.zero_run(False)
            else:
                self.activations = activations
            return None
        self.layer.register_forward_hook(forward_hook)

    def to(self, device):
        super().to(device)
        self.activations.to(device)
        self.zero_activations.to(device)
        self.target.to(device)

    def __call__(self):

        activations = self.activations
        b = activations.shape[0]

        c1 = int(self.targeted)
        c2 = int(self.anchored)
        loss = 0
        if c1:
            loss = loss + c1 * F.cosine_similarity(activations, self.target.repeat(b, 1))
        if c2:
            loss = loss + c2 * F.cosine_similarity(activations, self.zero_activations)
        return loss.mean()
    
class DistanceLoss(Loss):
    def __init__(self, device, eps, distance_fn, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.eps = eps
        self.distance_fn = distance_fn

    def zero_run(self, inputs, *args, **kwargs):
        self.zero_inputs = inputs.clone().detach()
    
    def __call__(self, inputs, *args, **kwargs):
        b = inputs.shape[0]
        loss = max(self.distance_fn(inputs, self.zero_inputs), self.eps) - self.eps
        return loss / b

class SumLoss(Loss):
    def __init__(self, losses, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.losses = losses

    def zero_run(self, flag):
        for loss in self.losses:
            loss.zero_run(flag)

    def __call__(self):
        sum_loss = 0
        for loss in self.losses:
            loss_i = loss().to(self.device)
            sum_loss = sum_loss + loss_i
        return sum_loss
    
    def to(self, device):
        super().to(device)
        for loss in self.losses:
            loss.to(device)