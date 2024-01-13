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
train_gt = sorted(glob(PATH + '*.txt')) 
train_img = [s[:-4] + '.jpg' for s in train_gt] # change .txt to .jpg

''' 
device and param set up 
'''
torch.backends.cudnn.deterministic = True # reproducibility 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
batch_size = 1
save_location = "results/"

dataset = AnnotateData(images=train_img, annotations=train_gt, gamma=False, test=True)  
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=False)

''' 
load pre-trained model 
'''
size = 512
model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).to(device) # two classes for snow and background 
model = custom_setup(size, model) 
model.load_state_dict(torch.load("pretrained_weights/SSD_trained_weights-2.pt", map_location=device), strict=False)  

count = 0  
threshold = 0.8# confidence score 
''' 
inference 
''' 
model.to(device).eval()
with torch.no_grad(): 
    for rgb_img in tqdm(loader): 
        rgb_img.to(device) 
        
        with torch.no_grad():
            result = model(rgb_img.to(device))

        annotation = result[0]["boxes"].cpu().numpy() # N x 4 
        score = result[0]["scores"].cpu().numpy() # N 
        print(np.mean(score))

        # Thresholding 
        annotation = annotation[score >= threshold,:] 
        print(annotation.shape)
        score = score[score >= threshold] 

        # dimension N x C x H x W -> H x W x C
        rgb_img = rgb_img[0,:,:,:].permute(1, 2, 0) 
        rgb_img *= 255. # undo normalization 
        rgb_img = rgb_img.cpu().numpy()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) 

        # draw bounding boundaries 
        for idx, box in enumerate(annotation): 
            # box.shape [4], idx 
            ymin = int(box[1] * rgb_img.shape[0] / rgb_img.shape[0])
            ymax = int(box[3] * rgb_img.shape[0] / rgb_img.shape[0])
            xmin = int(box[0] * rgb_img.shape[1] / rgb_img.shape[1])
            xmax = int(box[2] * rgb_img.shape[1] / rgb_img.shape[1])
            cv2.rectangle(rgb_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(rgb_img, "SNOW", (xmin, ymin-5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 0), 2, lineType=cv2.LINE_AA)  

        cv2.imwrite(save_location + f"results-{count}.jpg", rgb_img)

        # store result as numpy array 
        np.save(save_location + f"results-{count}.npy", result)  
        # annotations = np.load(save_location + "results-{count}.npy")  
        count += 1




