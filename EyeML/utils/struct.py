# Dataset 
import torch 
from torch.utils.data import Dataset 
from utils.tools import * 

class AnnotateData(Dataset): 
    '''Dataloader for annotated data  
    Returns an iterable class for dataloader
    '''
    def __init__(self, images, annotations=None, gamma=False, test=False): 
        '''param 2D image to annotate, gt annotations, gamma, test mode'''
        self.images = images 
        self.gamma = gamma 
        self.test = test 
        self.annotations = annotations 

    def __getitem__(self, idx): 
        '''param index of image to access 
        Returns a pair of image and gt annotation for training and image and name of image for testing'''
        print(self.images[idx])
        rgb_img = img_read(self.images[idx]) # numpy array of image 

        # perform gamma correction 
        if self.gamma: 
            rgb_img = rgb_img**(1/0.6) 

        # convert numpy to torch 
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        # read annotations 
        box_gt = annotation_read(self.annotations[idx]) 

        # convert numpy to torch 
        box_gt = torch.from_numpy(box_gt) 

        # convert from yolo to bbox 
        box_gt = yolo_to_bbox_torch(box_gt) 

        target = {}
        target["boxes"] = box_gt[:,1:]
        target["labels"] = box_gt[:,0].type(torch.LongTensor)

        # inference 
        if self.test: 
            return rgb_img # add rgb_name here as self.images[idx] 
         
        return rgb_img, target

    def __len__(self): 
        '''Returns the number of data points'''
        assert(len(self.images) == len(self.annotations))
        return len(self.images)  

