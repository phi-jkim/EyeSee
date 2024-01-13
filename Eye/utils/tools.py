import numpy as np 
import cv2 
import torch 

from torchvision.models.detection import _utils
from torchvision.models.detection.ssd import SSDClassificationHead

def img_read(file): 
    '''param file path 
    Returns image as a numpy array (H x W x 3)''' 
    # rgb_img = np.asarray(Image.open(file)) # load path and then convert to numpy 
    # print(rgb_img.shape) 

    image = cv2.imread(file)  
    # cv2.imshow('Image', image) # BGR show 

    # convert the image to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # nomralize image 
    image = (image / 255.).astype(np.float32) 

    return image 

def annotation_read(file): 
    '''param file path 
    Returns annotation as a torch tensor (N x 5)''' 

    annotation = open(file, 'r') 
    data = annotation.readlines() 
    annotation.close() 

    boxes = []
    for box in data: 
        c, x, y, w, h = map(float, box.split(' '))
        boxes.append([c, x, y, w, h]) 
    boxes = np.array(boxes)  
    return boxes 

def yolo_to_bbox_torch(yolo_tensor):
    '''param yolo tensor 
    Returns the yolo tensor in bbox format'''
    labels = yolo_tensor[:, 0].unsqueeze(1)

    x_center = yolo_tensor[:, 1]
    y_center = yolo_tensor[:, 2]
    width = yolo_tensor[:, 3]
    height = yolo_tensor[:, 4]

    x_min = (x_center - width / 2).unsqueeze(1)
    y_min = (y_center - height / 2).unsqueeze(1)
    x_max = (x_center + width / 2).unsqueeze(1)
    y_max = (y_center + height / 2).unsqueeze(1)

    bbox_tensor = torch.cat((labels, x_min, y_min, x_max, y_max), dim=1)
    return bbox_tensor

def custom_collate(batch):
    '''param batch of data from dataloader
    Returns images and targets by zipping and then stacking'''
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    return images, targets

def custom_setup(size, model): 
    '''param size of desired image input for raining, model for training
    Returns model with updated head'''
    # parameter numbers 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # change classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=2, # num_classes 
    )

    # modify to image size 
    model.transform.min_size = (size,)
    model.transform.max_size = size

    return model 