import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def get_logits(model, x_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        output = model(x.cuda())
    
    return output.cpu().numpy()

def get_predictions(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)
    with torch.no_grad():
        output = model(x.cuda())
    
    return (output.cpu().max(dim=-1)[1] == y).numpy()

def get_predictions_and_gradients(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)

    with torch.enable_grad():
        output = model(x.cuda())
        loss = nn.CrossEntropyLoss()(output, y.cuda())

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()

    return pred, grad

