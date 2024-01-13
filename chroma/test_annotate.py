# general util imports 
from glob import glob 
import os 

# torch, ml 
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm 

# torchvision 
import torchvision.models.detection as detection

# struct dataloader 
from utils.struct import AnnotateData
from utils.tools import * 

# For MAC fix library issue 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
PATH = "data/" 
train_img = sorted(glob(PATH + '*.jpg'))
train_gt = sorted(glob(PATH + '*.txt')) 

''' 
device and param set up 
'''
torch.backends.cudnn.deterministic = True # reproducibility 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
batch_size = 1
save_location = "results/"

dataset = AnnotateData(images=train_img, annotations=train_gt, gamma=False, test=True)  
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                    pin_memory=True, drop_last=False)

''' 
load pre-trained model 
'''
size = 320 
model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT) # two classes for snow and background 
model = custom_setup(size, model) 
model.load_state_dict(torch.load("pretrained_weights/YOLO_trained_weights-100.pt"), strict=False)  
count = 0  

''' 
inference 
''' 
model.to(device).eval()
with torch.no_grad(): 
    for rgb_img in tqdm(loader): 
        rgb_img.to(device) 
        
        with torch.no_grad():
            result = model(rgb_img.to(device))

        print(result[0]["boxes"], result[0]["scores"])
        # outpts[0]["boxes"], outpts[0]["scores"]
        # check if greater than threshold 

        # store result as numpy array 
        np.save(save_location + f"results-{count}.npy", result)  
        # annotations = np.load(save_location + "results-{count}.npy")  
        count += 1




