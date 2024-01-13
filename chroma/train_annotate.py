# general util imports 
from glob import glob 
import os 

# torch, ml 
import torch 
from torch.utils.data import DataLoader
import torchvision.models.detection as detection
from tqdm import tqdm 

# struct dataloader 
from utils.struct import AnnotateData
from utils.tools import * 


PATH = "data/" 
train_img = sorted(glob(PATH + '*.jpg'))
train_gt = sorted(glob(PATH + '*.txt')) 

''' 
device and param set up 
'''
torch.backends.cudnn.deterministic = True # reproducibility 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-4 
learning_rate_decay = 1e-6
epochs = 100 
batch_size = 6
save_freq = 10 
save_location = "./pretrained_weights/"

# load data as dataloader 
dataset = AnnotateData(images=train_img, annotations=train_gt, gamma=False, test=False)  
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                    pin_memory=True, drop_last=True, collate_fn=custom_collate)

'''
set up pre-trained model and set output size as 320 
'''
size = 320 
model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT) # two classes for snow and background 
model = custom_setup(size, model) 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay) 
epoch_loss = 0 

''' 
training 
''' 
for epoch in range(epochs): 
    torch.cuda.empty_cache() 
    model.train() 
    
    for rgb_img, targets in tqdm(loader): 
        optimizer.zero_grad()

        rgb_img = rgb_img.to(device) 

        # result 
        targets = [{keys: value.to(device) for keys, value in target.items()} for target in targets]
        epoch_loss = model(rgb_img, targets)  

        losses = sum(loss for loss in epoch_loss.values()) 

        # loss compute 
        losses.backward() 

        # update gradients 
        optimizer.step() 
    
    if epoch % save_freq == 0: 
        print("{epoch} loss is {epoch_loss}".format(epoch = epoch, epoch_loss = epoch_loss))
    
    # checkpoint 
    if epoch == epochs-1: 
        torch.save(model.state_dict(), os.path.join(save_location, f'YOLO_trained_weights-{epoch+1}.pt')) 