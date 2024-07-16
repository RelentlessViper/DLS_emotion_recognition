import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 14 * 14, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc(x))
        return x
    
class Embedding_generator():
    def __init__(self, path_to_weights: str = '../Models/encoder_v1.pth', latent_dim=128) -> None:
        self.latent_dim = latent_dim
        
        self.model = Encoder(latent_dim=128)
        self.model.load_state_dict(torch.load(path_to_weights))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.model(image)

        return embedding.squeeze().numpy()