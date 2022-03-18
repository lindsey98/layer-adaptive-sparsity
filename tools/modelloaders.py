import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from tools.models import *
from tools.pruners import prune_weights_reparam
import torch

def model_and_opt_loader(model_string,DEVICE,weights_path=None):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet101':
        model = ResNet101().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'resnet152':
        model = ResNet152().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'resnet152v2':
        model = ResNet152V2().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'resnet152v3':
        model = ResNet152V3().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'resnet101v2':
        model = ResNet101V2().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'resnet101v3':
        model = ResNet101V3().to(DEVICE)
        amount = 0.20
        batch_size = 128
        opt_pre = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 50,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, momentum=0.9, weight_decay=0.001),
            "steps": 40,
            "scheduler": None
        }
    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 60000,
            "scheduler": None
        }
    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    else:
        raise ValueError('Unknown model')

    """ Load (IF NEEDED) """
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=torch.device(DEVICE))['net']
        model.load_state_dict(state_dict)
    prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post