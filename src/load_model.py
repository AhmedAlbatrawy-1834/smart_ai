import torch
from torchvision import models
import torch.nn as nn

def train_model(path, class_names, device):
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    model.load_state_dict(torch.load(path, map_location=device))

    model = model.to(device)
    model.eval()
    return model