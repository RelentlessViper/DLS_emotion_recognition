from PIL import Image

import torch
from torchvision.transforms import v2

class InputTransform:
    
    def __init__(self, size=(112,112)):
        self.size = size
        
    def transform(self, orig_img):
        gray_img = v2.Grayscale()(orig_img)
        resized_img = v2.Resize(size=self.size)(gray_img)
        
        return resized_img
    
    
if (__name__ == '__main__'):
    with Image.open(r'Input_Transform\test.png') as im:
        img = InputTransform().transform(im)
        img.show()
    with Image.open(r'Input_Transform\test colored.png') as im:
        img = InputTransform().transform(im)
        img.show()