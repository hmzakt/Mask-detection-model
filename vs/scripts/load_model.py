import torch
from model import ResNet18Clone

def load_trained_model(model_path = "model/face_detection_model.pth"):
    model = ResNet18Clone(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

if __name__ == "__main__":
    model = load_trained_model
    print("model loaded")