import Eye
from CameraTest import * 
import torch
import torchvision.models.detection as detection
import numpy as np 

# for storing 
# import cv2

''' 
device and param set up 
'''
torch.backends.cudnn.deterministic = True # reproducibility 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
batch_size = 1
save_location = "/home/bread/Desktop/results"

''' 
load pre-trained model 
'''
size = 512
model = detection.ssdlite320_mobilenet_v3_large(weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).to(device) # two classes for snow and background 
model = custom_setup(size, model) 
model.load_state_dict(torch.load("pretrained_weights/SSD_trained_weights-2.pt", map_location=device), strict=False)  

threshold = 0.80# confidence score 
model.to(device).eval()

'''
initialize camera 
'''

cam0 = MyCamera(0, "/home/bread/Desktop/TestPics", "RGB")
cam1 = MyCamera(1, "/home/bread/Desktop/TestPics", "IR")

# initialize config 
cam0.create_config()
cam1.create_config()

# align configuration 
cam0.align_configuration(cam0.config) # RGB camera 
cam1.align_configuration(cam0.config) # IR camera 

# configure cameras 
cam0.configure(cam0.config)
cam1.configure(cam1.config)

# start cameras 
cam0.start()
count = 0

# initialize buzzer 
buzzer = Buzzer(21)

while True: 
    rgb_image = cam0.capture_array() # H x W x 3 
    rgb_image = (np.array(rgb_image) / 255.).astype(np.float32) 
    # 
    rgb_image = torch.from_numpy(rgb_image.permute(2, 0, 1)).unsqueeze(0)# fix dimension 

    with torch.no_grad(): 
        result =  model(rgb_image.to(device))

    annotation = result[0]["boxes"].cpu().numpy() # N x 4 
    score = result[0]["scores"].cpu().numpy() # N 

    # Thresholding 
    annotation = annotation[score >= threshold,:] 
    score = score[score >= threshold] 

    # dimension N x C x H x W -> H x W x C
    rgb_image = rgb_image[0,:,:,:].permute(1, 2, 0) 
    rgb_image *= 255. # undo normalization 
    rgb_image = rgb_image.cpu().numpy()
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 

    for idx, box in enumerate(annotation): 
        # y coordinates of box 
        ymin = int(box[1] * rgb_image.shape[0] / rgb_image.shape[0])
        ymax = int(box[3] * rgb_image.shape[0] / rgb_image.shape[0])
        # x coordinates of box 
        xmin = int(box[0] * rgb_image.shape[1] / rgb_image.shape[1])
        xmax = int(box[2] * rgb_image.shape[1] / rgb_image.shape[1])
        # draw rectangle and save 
        # cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        # cv2.putText(rgb_image, "SNOW", (xmin, ymin-5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
        #             (0, 255, 0), 2, lineType=cv2.LINE_AA)  
        
        '''buzzer'''
        cy = (ymin + ymax) / 2 
        cx = (xmin + xmax) / 2

        buzzer_interval = cy / 512
        if buzzer_interval < 0.2:
            buzzer_interval = 0.2

        buzzer.beep(buzzer_interval)
        time.sleep(max(0, 0.5-buffer_interval)) 
        
    # cv2.imwrite(save_location + f"results-{count}.jpg", rgb_img)
    count += 1


