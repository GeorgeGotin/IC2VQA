import numpy as np
import torch
from torch import nn
import torchvision

import Loss
import Mutations

from Models import get_model

class Attack(object):
    def __init__(self, devices=['cuda'], **kwargs):
        self.devices = devices
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def forward(self, inputs):
        self.loss.zero_run(inputs)

        for i in range(self.epoch):
            loss = self.loss(inputs)
            loss.backward()
    
    def change_config(self, config):
        for key, value in config.items():
            self.__setattr__(key, value)

    def __call__(self, input):
        return self.forward(input)

class I2V(Attack):
    def __init__(self, devices=['cuda'], epoch = 1, eps = 5/255, alpha = 1/255, models  = [('nima',None)], order = 'conticous'):
        super().__init__(devices, epoch, eps, alpha, models, order)
        assert len(devices) >= len(self.models)
        losses = []
        self.models = []
        for i in range(len(self.models)):
            model_name = self.models[i][0]
            layer_names = self.models[i][1]
            if not isinstance(layer_names, list):
                layer_names = [layer_names]
            device = devices[i]
            model, preprocces = get_model(model_name, device)
            self.models.append((model, preprocces))
            for layer_name in layer_names:
                if layer_name is None:
                    layer = self.get_default_layer(model_name)
                else:
                    layer = self.get_layer(model, layer_name)
                losses.append(Loss.CrossLayerLoss(layer, device))
        self.loss = Loss.SumLoss(losses, device='cpu')

    def get_default_layer(self, model_name):
        if (model_name == 'nima'):
            return self.get_layer('net.global_pool')
        if model_name == 'paq2piq':
            return self.get_layer('net.body')
        return self.model
    
    def get_layer(self, model, layer_name):
        last_name = ''
        for name in layer_name.split('.'):
            last_name = name
            model = getattr(model, name)
        self.public_name += f'.{last_name}'
        return model
    
    def forward(self, inputs):
        zero_device = inputs.device
        mutation = Mutations.Adv(inputs, self.eps, 1/255, self.devices[0])
        self.loss.zero_run()
        for model, preprocces in self.models:
            model(preprocces(inputs))
        self.loss.zero_run(False)
        optim = torch.optim.Adam(mutation.parameters(), lr=self.alpha)
        for idx in range(self.epoch):
            optim.zero_grad()
            for model, preprocces in self.models:
                model(preprocces(mutation.mutate(inputs)))
            loss = self.loss()
            loss.backward()
            optim.step()
        return mutation.mutate(inputs).to(zero_device)
    
def itterational_attack(inputs, attack, batch_size=30):
    length = inputs.shape[0]
    indexes = [(i, min(length, i + batch_size)) for i in range(0, length, batch_size)]
    for i, j in indexes:
        #print(i, j)
        inputs[i:j] = attack(inputs[i:j]).to('cpu')
    return inputs