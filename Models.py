import torch
import torchvision

import pyiqa
from torchvision.transforms import v2, InterpolationMode
# from .SPAQ.BL_demo import Demo
# from linearity_api import LinearityIQA
import clip

def get_model(imetric_name, device='cuda'):
    preprocces = lambda x: x
    if imetric_name.startswith('spaq'):  
        imetric = Demo(config=False, device=device).model
        imetric.device = device
        imetric.lower_better = False
        imetric.eval()
        imetric.range = 100
    if imetric_name.startswith('linearity'):
        imetric = LinearityIQA(device=device)
        imetric.lower_better = False
        imetric.eval()
        imetric.device = device
        imetric.range = 100
    if imetric_name.startswith('nima'):
        imetric = pyiqa.create_metric('nima', as_loss = not imetric_name.endswith('noloss'), device=device)
        imetric.eval()
        imetric.range = 10
    if imetric_name.startswith('paq2piq'):
        imetric = pyiqa.create_metric('paq2piq', as_loss = not imetric_name.endswith('noloss'), device=device)
        imetric.eval()
        imetric.range = 100
    if imetric_name.startswith('resnet101'):
        imetric = torchvision.models.get_model(imetric_name, weights='DEFAULT').to(device)
        imetric.device = device
        imetric.eval()
    if imetric_name.startswith('clip'):
        model,_ = clip.load('ViT-B/32', device=device)
        x = model.visual.input_resolution
        preprocces = v2.Compose([
            v2.Resize(x, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(x),
            v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            v2.ToDtype(model.dtype)
        ])
        imetric = model.visual
        imetric.eval()
        imetric.device = device
    return imetric, preprocces