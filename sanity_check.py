import torch 
import numpy as np 
import torchvision 

image = torch.from_numpy(np.random.random((1, 3, 800, 800)).astype(np.float32)) 
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large() 

target = {} 
target["boxes"] = torch.from_numpy(np.random.random((10, 4)).astype(np.float32)) 
target["labels"] = torch.from_numpy(np.random.random((10)).astype(np.float32)) 

'''
batch 
image: N x C x H x W 
target: array of dictionaries 
'''
outpt = model(image, [target]) 

'''
testing mode 
'''

model.eval()
model(image)
