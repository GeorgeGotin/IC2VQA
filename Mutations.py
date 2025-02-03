import torch
import numpy as np

from torch.autograd import Variable

class Mutation:
    def __init__(self, inputs, alpha = 1/255, device='cuda'):
        self.alpha = alpha
        self.device = device

    def parameters(self):
        return ()
    
    def mutate(self, inputs):
        return inputs
    
class Adv(Mutation):
    def __init__(self, inputs, eps=5/255,alpha = 1/255, device='cuda'):
        super().__init__(inputs, alpha, device)
        self.adv = Variable(torch.ones_like(inputs).to(self.device)*alpha, requires_grad=True)
        self.eps = eps

    def parameters(self):
        return (self.adv,)
    
    def mutate(self, inputs):
        res = torch.clip(inputs.to(self.device) + torch.clip(self.adv, -self.eps, self.eps), 0, 1)
        return res