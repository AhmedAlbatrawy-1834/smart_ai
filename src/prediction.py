import torch

def predict(img, model, class_names):
    with torch.no_grad():
        output = model(img)
        _, predicted_idx = torch.max(output, 1)
        
    return class_names[predicted_idx.item()]